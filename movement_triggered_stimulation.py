"""Movement-triggered random-amplitude stimulation server.

pyControl owns trial timing and sends a trigger code when movement qualifies.
This process owns stimulation condition randomization, hardware timing, and the
returned event markers used by pyControl for behavior/video alignment.
"""

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir
sys.path.append(root_dir)

from matlab_stimulator import MatlabStimulator
from utils.loader import CONFIG
from utils.serial_port_resolver import resolve_serial_port
from utils.stim_session import (
    build_event_fields,
    make_event_logger,
    normalize_pulse_mode,
    resolve_channel_metadata,
)


HW_CONF = CONFIG["hardware"]
CALIBRATION_DIR = CONFIG["hardware"].get("calibration_dir")
RANDOMIZATION_MODE = "balanced_shuffle"
DEFAULT_TRIGGER_CODE = 1
SESSION_MARKER_CODE = 0
DEFAULT_PYCONTROL_CODES = {
    "session_start": 110,
    "session_end": 111,
    "stim_on": 101,
    "stim_off": 102,
    "stim_pulse": 103,
    "stim_sham": 104,
}


@dataclass(frozen=True)
class MovementTriggeredCondition:
    """One validated movement-triggered stimulation condition."""

    name: str
    amp_values: list[float]
    sham: bool
    raw: dict


class BalancedConditionDeck:
    """Shuffle all condition indices, consume them once, then reshuffle."""

    def __init__(self, condition_count, rng):
        if condition_count <= 0:
            raise ValueError("condition_count must be greater than 0.")
        self.condition_count = condition_count
        self.rng = rng
        self.cycle = 0
        self.deck = []

    def next_index(self):
        if not self.deck:
            self.deck = list(range(self.condition_count))
            self.rng.shuffle(self.deck)
            self.cycle += 1
        return self.deck.pop(0), self.cycle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pyControl-triggered random-amplitude train stimulation."
    )
    parser.add_argument(
        "--protocol",
        default=None,
        help=(
            "Movement-triggered protocol name. Self-contained experiment configs "
            "use [experiment].name; legacy configs use [movement_triggered_protocols]."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode, overriding [run].mock_mode.",
    )
    mode.add_argument(
        "--real",
        action="store_true",
        help="Use real stimulator hardware. Requires protocol real_enabled = true.",
    )
    parser.add_argument(
        "--trigger-port",
        default=None,
        help="Serial port for pyControl triggers. Defaults to config.toml.",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=None,
        help="Trigger-link baud rate. Defaults to [hardware].baud_rate.",
    )
    parser.add_argument(
        "--max-triggers",
        type=int,
        default=None,
        help="Stop after this many accepted movement stimulation triggers.",
    )
    parser.add_argument(
        "--runtime",
        type=float,
        default=None,
        help="Stop once this many seconds have elapsed from session start.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for balanced condition shuffling. Generated if omitted.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output; event details are still written to the CSV log.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Session ID used for the local stimulation event log. With --animal, "
            "this is the behavior/ephys session folder name."
        ),
    )
    parser.add_argument(
        "--animal",
        default=None,
        help="Animal ID used to place logs under <data_root>/<animal>/<session_id>/.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root folder for animal/session logs. Defaults to [logging].data_root.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for stimulation event logs. Defaults to [logging].stim_event_dir.",
    )
    parser.add_argument(
        "--no-local-log",
        action="store_true",
        help="Disable the local stimulation CSV/JSON event log.",
    )
    args = parser.parse_args()
    if args.max_triggers is not None and args.max_triggers <= 0:
        parser.error("--max-triggers must be greater than 0.")
    if args.runtime is not None and args.runtime <= 0:
        parser.error("--runtime must be greater than 0 seconds.")
    return args


def normalize_stim_values(values, channel_count, name):
    """Return one finite numeric value per configured stimulation channel."""

    if isinstance(values, (int, float)):
        normalized = [float(values)] * channel_count
    else:
        try:
            normalized = [float(value) for value in values]
        except TypeError as exc:
            raise ValueError(f"{name} must be a scalar or list of numbers.") from exc
        if len(normalized) == 1:
            normalized = normalized * channel_count
        elif len(normalized) != channel_count:
            raise ValueError(
                f"{name} must be a scalar or have one value per configured channel."
            )

    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} values must be finite numbers.")
    return normalized


def validate_amp_values(amp_values, max_amp_values, channels):
    for channel, value, max_value in zip(channels, amp_values, max_amp_values):
        if value < 0:
            raise ValueError("Amplitude must be greater than or equal to 0 mA.")
        if value > max_value + 1e-12:
            raise ValueError(
                f"Amplitude {value:g} mA for channel {channel} exceeds the "
                f"profile maximum of {max_value:g} mA."
            )


def validate_duration(duration_s):
    try:
        duration_s = float(duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("Movement-triggered protocol requires duration.") from exc
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration must be a finite value greater than 0 seconds.")
    return duration_s


def validate_trigger_code(trigger_code):
    try:
        trigger_code = int(trigger_code)
    except (TypeError, ValueError) as exc:
        raise ValueError("trigger_code must be an integer.") from exc
    if trigger_code == SESSION_MARKER_CODE:
        raise ValueError("trigger_code 0 is reserved for session markers.")
    if trigger_code < 0 or trigger_code > 65535:
        raise ValueError("trigger_code must fit in an unsigned 16-bit integer.")
    return trigger_code


def self_contained_protocol():
    experiment = CONFIG.get("experiment", {})
    stim = CONFIG.get("stim")
    conditions = CONFIG.get("conditions")
    if not experiment and stim is None and conditions is None:
        return None

    experiment_type = experiment.get("type")
    if experiment_type not in (None, "movement_triggered"):
        return None
    if stim is None and conditions is None:
        return None

    protocol_name = experiment.get("name", "movement_triggered")
    protocol = dict(experiment)
    protocol["stim"] = stim
    protocol["conditions"] = conditions or []
    return protocol_name, protocol


def resolve_protocol(protocol_name=None):
    direct_protocol = self_contained_protocol()
    if direct_protocol is not None:
        selected, protocol = direct_protocol
        if protocol_name and protocol_name != selected:
            raise ValueError(
                f"Self-contained movement-triggered config is named '{selected}', "
                f"but --protocol requested '{protocol_name}'."
            )
        return selected, protocol

    protocols = CONFIG.get("movement_triggered_protocols", {})
    selected = protocol_name or CONFIG.get("run", {}).get(
        "active_movement_triggered_protocol"
    )
    if not selected:
        available = ", ".join(sorted(protocols)) or "(none configured)"
        raise ValueError(
            "No movement-triggered protocol selected. Set "
            "[run].active_movement_triggered_protocol or pass --protocol. "
            f"Available protocols: {available}"
        )
    if selected not in protocols:
        available = ", ".join(sorted(protocols)) or "(none configured)"
        raise ValueError(
            f"Unknown movement-triggered protocol '{selected}'. "
            f"Available protocols: {available}"
        )
    return selected, protocols[selected]


def pycontrol_codes():
    event_conf = CONFIG.get("pycontrol_events", {})
    configured = event_conf.get("codes", {})
    return {**DEFAULT_PYCONTROL_CODES, **configured}


def resolve_stim_profile(protocol_name, protocol):
    if protocol.get("stim") is not None:
        stim_profile = dict(protocol["stim"])
        profile_name = protocol.get("profile") or protocol_name
        if "max_amp" in stim_profile:
            if "amp" in stim_profile and stim_profile["amp"] != stim_profile["max_amp"]:
                raise ValueError(
                    "Use either [stim].max_amp or [stim].amp for the amplitude "
                    "safety ceiling, not both."
                )
            stim_profile["amp"] = stim_profile["max_amp"]
        if "amp" not in stim_profile:
            raise ValueError(
                f"Movement-triggered experiment '{protocol_name}' requires "
                "[stim].max_amp."
            )
        return profile_name, stim_profile

    profiles = CONFIG.get("stimulation", {})
    profile_name = protocol.get("profile")
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none configured)"
        raise ValueError(
            f"Movement-triggered protocol '{protocol_name}' refers to unknown "
            f"stimulation profile '{profile_name}'. Available profiles: {available}"
        )
    return profile_name, dict(profiles[profile_name])


def validate_stim_profile(protocol_name, stim_profile):
    for key in ("freq", "pw", "amp"):
        if key not in stim_profile:
            raise ValueError(
                f"Movement-triggered experiment '{protocol_name}' requires [stim].{key}."
            )


def validate_movement_triggered_protocol(protocol_name, protocol):
    """Validate config and return profile details plus normalized conditions."""

    profile_name, stim_profile = resolve_stim_profile(protocol_name, protocol)
    validate_stim_profile(protocol_name, stim_profile)
    pulse_mode = normalize_pulse_mode(stim_profile, validate=True)
    if pulse_mode != "train":
        raise ValueError(
            f"Movement-triggered protocol '{protocol_name}' requires a train-mode "
            f"profile; '{profile_name}' uses pulse_mode={pulse_mode!r}."
        )

    duration_s = validate_duration(protocol.get("duration"))
    trigger_code = validate_trigger_code(
        protocol.get("trigger_code", DEFAULT_TRIGGER_CODE)
    )
    channels = list(MatlabStimulator._normalize_channels(stim_profile.get("channel")))
    channel_count = len(channels)
    pw_values = normalize_stim_values(stim_profile["pw"], channel_count, "pw")
    if any(value < 0 for value in pw_values):
        raise ValueError("Profile pulse width values must be >= 0 ms.")
    max_amp_values = normalize_stim_values(stim_profile["amp"], channel_count, "amp")
    validate_amp_values(max_amp_values, max_amp_values, channels)

    raw_conditions = protocol.get("conditions", [])
    if not raw_conditions:
        raise ValueError(
            f"Movement-triggered protocol '{protocol_name}' must define at least one condition."
        )

    conditions = []
    for index, condition in enumerate(raw_conditions, start=1):
        name = str(condition.get("name") or f"condition_{index}")
        if "amp" not in condition:
            raise ValueError(f"Condition '{name}' is missing amp.")
        amp_values = normalize_stim_values(condition["amp"], channel_count, "amp")
        validate_amp_values(amp_values, max_amp_values, channels)
        sham = bool(condition.get("sham", False)) or all(
            value == 0 for value in amp_values
        )
        conditions.append(
            MovementTriggeredCondition(
                name=name,
                amp_values=amp_values,
                sham=sham,
                raw=dict(condition),
            )
        )

    return {
        "profile_name": profile_name,
        "stim_profile": stim_profile,
        "pulse_mode": pulse_mode,
        "duration_s": duration_s,
        "trigger_code": trigger_code,
        "channels": channels,
        "pw_values": pw_values,
        "max_amp_values": max_amp_values,
        "conditions": conditions,
    }


def resolve_trigger_port(args, hw_conf):
    if args.trigger_port:
        return args.trigger_port, [args.trigger_port], None
    return resolve_serial_port(
        hw_conf,
        "trigger_ports",
        "trigger_port",
        label="pyControl trigger port",
        require_available=False,
    )


def open_trigger_link(port, baud_rate):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed in this Python environment. "
            "Use the conda environment declared in env.yml or install pyserial."
        ) from exc

    return serial.Serial(port, baud_rate, timeout=0.01)


def send_pycontrol_marker(link, code):
    if link is None or code is None:
        return
    if not 0 <= int(code) <= 65535:
        raise ValueError(f"pyControl marker code {code} out of 16-bit range.")
    link.write(int(code).to_bytes(2, byteorder="little", signed=False))
    if hasattr(link, "flush"):
        link.flush()


def flush_trigger_buffer(link):
    """Drop queued triggers collected while a stimulation attempt was running."""

    if link is None:
        return
    if hasattr(link, "reset_input_buffer"):
        link.reset_input_buffer()
        return
    pending = getattr(link, "in_waiting", 0)
    if pending:
        link.read(pending)


def event_fields(
    protocol_name,
    profile_name,
    stim_profile,
    stim_port,
    pulse_mode,
    mock_mode,
    *,
    duration=None,
    command=None,
    command_code=None,
    pycontrol_code=None,
    status=None,
    amp_values=None,
    pw_values=None,
    **details,
):
    fields = build_event_fields(
        "movement_triggered",
        profile_name,
        stim_profile,
        stim_port,
        pulse_mode,
        mock_mode,
        experiment=protocol_name,
        duration=duration,
        command=command,
        command_code=command_code,
        pycontrol_code=pycontrol_code,
        status=status,
        amp_values=amp_values,
        pw_values=pw_values,
    )
    fields.update(details)
    return fields


def should_stop(accepted_triggers, max_triggers, session_start, runtime_s):
    if max_triggers is not None and accepted_triggers >= max_triggers:
        return True
    if runtime_s is not None and time.perf_counter() - session_start >= runtime_s:
        return True
    return False


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    run_conf = CONFIG.get("run", {})

    try:
        protocol_name, protocol = resolve_protocol(args.protocol)
        validated = validate_movement_triggered_protocol(protocol_name, protocol)
    except ValueError as exc:
        print(f"!! FATAL: {exc}")
        return 2

    mock_mode = bool(run_conf.get("mock_mode", True))
    if args.real:
        mock_mode = False
    elif args.mock:
        mock_mode = True

    if not mock_mode and not protocol.get("real_enabled", False):
        print(
            f"!! FATAL: Movement-triggered protocol '{protocol_name}' is not "
            "enabled for real hardware."
        )
        print(
            "!! Set real_enabled = true only after checking channels, amplitudes, "
            "and timing on the bench."
        )
        return 2

    try:
        stim_port, stim_port_candidates, detected_ports = resolve_serial_port(
            HW_CONF,
            "stimulator_ports",
            "stimulator_port",
            label="stimulator port",
            require_available=not mock_mode,
        )
        trigger_port, trigger_candidates, trigger_detected = resolve_trigger_port(
            args, HW_CONF
        )
    except RuntimeError as exc:
        print(f"!! FATAL: {exc}")
        return 2

    profile_name = validated["profile_name"]
    stim_profile = validated["stim_profile"]
    pulse_mode = validated["pulse_mode"]
    duration_s = validated["duration_s"]
    trigger_code = validated["trigger_code"]
    pw_values = validated["pw_values"]
    conditions = validated["conditions"]
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)
    deck = BalancedConditionDeck(len(conditions), rng)
    zero_values = [0.0] * len(validated["channels"])
    baud_rate = args.baud_rate or HW_CONF.get("baud_rate", 9600)
    codes = pycontrol_codes()

    matlab_path = os.path.join(root_dir, "matlab_backend")
    if not os.path.exists(matlab_path):
        print(f"!! ERROR: MATLAB backend not found at {matlab_path}")
        return 1

    channel_metadata = resolve_channel_metadata(stim_profile)
    log_conf = CONFIG.get("logging", {})
    log_enabled = bool(log_conf.get("stim_events_enabled", True)) and not args.no_local_log
    event_logger = make_event_logger(
        script="movement_triggered_stimulation.py",
        animal=args.animal,
        session_id=args.session_id,
        data_root=args.data_root,
        log_dir=args.log_dir,
        root_dir=root_dir,
        profile_name=profile_name,
        stim_profile=stim_profile,
        channel_metadata=channel_metadata,
        stim_port=stim_port,
        mock_mode=mock_mode,
        extra={
            "protocol_name": protocol_name,
            "protocol": protocol,
            "duration_s": duration_s,
            "trigger_code": trigger_code,
            "trigger_port": trigger_port,
            "baud_rate": baud_rate,
            "seed": seed,
            "randomization_mode": RANDOMIZATION_MODE,
            "conditions": [condition.raw for condition in conditions],
            "max_triggers": args.max_triggers,
            "runtime_s": args.runtime,
            "pycontrol_marker_codes": codes,
        },
        enabled=log_enabled,
    )

    link = None
    stim = None
    accepted_triggers = 0
    exit_code = 0
    session_start = time.perf_counter()

    def send_marker(marker):
        code = codes.get(marker)
        if code is None:
            return None
        try:
            send_pycontrol_marker(link, code)
        except Exception as exc:
            event_logger.record(
                "pycontrol_marker_send_failed",
                pycontrol_code=code,
                command=marker,
                status="pycontrol_send_failed",
                error=str(exc),
            )
            raise
        return code

    def run_attempt(command_code):
        nonlocal accepted_triggers

        condition_index, shuffle_cycle = deck.next_index()
        condition = conditions[condition_index]
        accepted_triggers += 1
        actual_onset_s = time.perf_counter() - session_start
        common_details = {
            "event_number": accepted_triggers,
            "condition_index": condition_index + 1,
            "shuffle_cycle": shuffle_cycle,
            "trigger_code": command_code,
            "actual_onset_s": actual_onset_s,
            "randomization_mode": RANDOMIZATION_MODE,
        }

        if condition.sham:
            if not args.quiet:
                print(
                    f">> {timestamp()} [Trigger {command_code}] "
                    f"event {accepted_triggers}: {condition.name} sham"
                )
            sham_code = send_marker("stim_sham")
            event_logger.record(
                "stim_sham",
                **event_fields(
                    protocol_name,
                    profile_name,
                    stim_profile,
                    stim_port,
                    pulse_mode,
                    mock_mode,
                    duration=duration_s,
                    command=condition.name,
                    command_code=command_code,
                    pycontrol_code=sham_code,
                    status="no_stimulation",
                    amp_values=condition.amp_values,
                    pw_values=pw_values,
                    **common_details,
                ),
            )
            time.sleep(duration_s)
            off_code = send_marker("stim_off")
            event_logger.record(
                "stim_off",
                **event_fields(
                    protocol_name,
                    profile_name,
                    stim_profile,
                    stim_port,
                    pulse_mode,
                    mock_mode,
                    duration=duration_s,
                    command=condition.name,
                    command_code=command_code,
                    pycontrol_code=off_code,
                    status="sent",
                    amp_values=zero_values,
                    pw_values=pw_values,
                    **common_details,
                ),
            )
            return

        if not args.quiet:
            print(
                f">> {timestamp()} [Trigger {command_code}] event {accepted_triggers}: "
                f"{condition.name} amp={condition.amp_values}mA duration={duration_s:g}s"
            )
        event_logger.record(
            "stim_on_request",
            **event_fields(
                protocol_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                duration=duration_s,
                command=condition.name,
                command_code=command_code,
                status="requested",
                amp_values=condition.amp_values,
                pw_values=pw_values,
                **common_details,
            ),
        )
        stim.stimulate(pw=pw_values, amp=condition.amp_values)
        on_code = send_marker("stim_on")
        event_logger.record(
            "stim_on",
            **event_fields(
                protocol_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                duration=duration_s,
                command=condition.name,
                command_code=command_code,
                pycontrol_code=on_code,
                status="sent",
                amp_values=condition.amp_values,
                pw_values=pw_values,
                **common_details,
            ),
        )
        try:
            time.sleep(duration_s)
        finally:
            stim.stop()
            off_code = send_marker("stim_off")
            event_logger.record(
                "stim_off",
                **event_fields(
                    protocol_name,
                    profile_name,
                    stim_profile,
                    stim_port,
                    pulse_mode,
                    mock_mode,
                    duration=duration_s,
                    command=condition.name,
                    command_code=command_code,
                    pycontrol_code=off_code,
                    status="sent",
                    amp_values=zero_values,
                    pw_values=pw_values,
                    **common_details,
                ),
            )

    print("=== Movement-Triggered Stimulation Server ===")
    print(f">> Protocol: {protocol_name}")
    print(f">> Stim config: {profile_name}")
    print(f">> Trigger code: {trigger_code}")
    print(f">> Duration: {duration_s:g}s")
    print(f">> Randomization: {RANDOMIZATION_MODE}, seed={seed}")
    print(
        f">> Stimulation settings: {stim_profile['freq']}Hz, "
        f"pw={pw_values}ms, pulse_mode={pulse_mode}"
    )
    print(f">> Mode: {'MOCK' if mock_mode else 'REAL'}")
    print(f">> Stimulator port: {stim_port}")
    print(f">> Trigger port: {trigger_port} @ {baud_rate}")
    if len(stim_port_candidates) > 1:
        print(f">> Stimulator port candidates: {stim_port_candidates}")
        if detected_ports is not None:
            print(f">> Detected serial ports: {detected_ports}")
    if len(trigger_candidates) > 1:
        print(f">> Trigger port candidates: {trigger_candidates}")
        if trigger_detected is not None:
            print(f">> Detected serial ports: {trigger_detected}")
    print(">> Conditions:")
    for index, condition in enumerate(conditions, start=1):
        label = "sham" if condition.sham else "stim"
        print(f"   {index}: {condition.name} ({label}) amp={condition.amp_values}mA")
    if event_logger.enabled:
        print(f">> Stimulation event log: {event_logger.csv_path}")

    try:
        event_logger.record(
            "session_start",
            **event_fields(
                protocol_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                duration=duration_s,
                amp_values=validated["max_amp_values"],
                pw_values=pw_values,
                trigger_code=trigger_code,
                seed=seed,
                randomization_mode=RANDOMIZATION_MODE,
            ),
        )

        stim = MatlabStimulator(
            matlab_path,
            mock_mode=mock_mode,
            calibration_dir=CALIBRATION_DIR,
            quiet=args.quiet,
        )
        print(">> Configuring Stimulator...")
        stim.configure(
            port=stim_port,
            freq=stim_profile["freq"],
            pw=stim_profile["pw"],
            amp=stim_profile["amp"],
            channels=stim_profile.get("channel"),
            inter_phase=stim_profile.get("inter_phase", 50e-6),
            pulse_mode=pulse_mode,
            single_pulse_train_ms=stim_profile.get("single_pulse_train_ms"),
        )
        print(">> Connecting...")
        stim.connect()

        print(f">> Opening Trigger Link on {trigger_port}...")
        link = open_trigger_link(trigger_port, baud_rate)
        print(">> Link Open. Waiting for movement triggers...")

        while not should_stop(
            accepted_triggers, args.max_triggers, session_start, args.runtime
        ):
            if getattr(link, "in_waiting", 0) >= 2:
                raw_data = link.read(2)
                if len(raw_data) != 2:
                    event_logger.record(
                        "trigger_malformed",
                        **event_fields(
                            protocol_name,
                            profile_name,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            status="ignored",
                        ),
                    )
                    continue

                command_code = int.from_bytes(raw_data, byteorder="little", signed=False)
                event_logger.record(
                    "trigger_received",
                    **event_fields(
                        protocol_name,
                        profile_name,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        duration=duration_s,
                        amp_values=validated["max_amp_values"],
                        pw_values=pw_values,
                        command_code=command_code,
                        status="received",
                    ),
                )

                if command_code == SESSION_MARKER_CODE:
                    if not args.quiet:
                        print(f">> {timestamp()} [Trigger 0] Session marker.")
                    event_logger.record(
                        "session_marker_received",
                        **event_fields(
                            protocol_name,
                            profile_name,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            command_code=command_code,
                            status="received",
                        ),
                    )
                    continue

                if command_code != trigger_code:
                    print(f">> [Trigger {command_code}] Unknown command ignored.")
                    event_logger.record(
                        "trigger_unknown",
                        **event_fields(
                            protocol_name,
                            profile_name,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            command_code=command_code,
                            status="ignored",
                        ),
                    )
                    continue

                try:
                    run_attempt(command_code)
                finally:
                    flush_trigger_buffer(link)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n>> Server stopping by user request.")
    except Exception as exc:
        print(f"\n!! Runtime Error: {exc}")
        event_logger.record(
            "runtime_error",
            error=str(exc),
            **event_fields(
                protocol_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                status="failed",
            ),
        )
        exit_code = 1
    finally:
        print(">> Shutting down...")
        if link is not None:
            link.close()
        if stim is not None:
            try:
                stim.stop()
            finally:
                stim.close()
        event_logger.record(
            "session_end",
            **event_fields(
                protocol_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                duration=duration_s,
                amp_values=validated["max_amp_values"],
                pw_values=pw_values,
                events_completed=accepted_triggers,
            ),
        )
        event_logger.close()
        print("Bye.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
