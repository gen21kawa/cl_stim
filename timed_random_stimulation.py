"""Timed random stimulation runner.

This script starts train-mode stimulation on a fixed onset-to-onset cadence and
chooses each event's amplitude/duration from a configured condition list. The
condition deck is shuffled, consumed once, then reshuffled so sham/stim counts
stay balanced over each cycle.
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
from utils.pycontrol_event_link import PyControlEventLink
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
DEFAULT_PYCONTROL_CODES = {
    "session_start": 110,
    "session_end": 111,
    "stim_on": 101,
    "stim_off": 102,
    "stim_pulse": 103,
    "stim_sham": 104,
}


@dataclass(frozen=True)
class TimedRandomCondition:
    """One validated timed-random condition."""

    name: str
    amp_values: list
    duration_s: float
    sham: bool
    raw: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed-interval random train stimulation from config.toml."
    )
    parser.add_argument(
        "--protocol",
        default=None,
        help=(
            "Protocol name from [timed_random_protocols]. Defaults to "
            "[run].active_timed_random_protocol."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Override the protocol onset-to-onset interval in seconds.",
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
    pycontrol = parser.add_mutually_exclusive_group()
    pycontrol.add_argument(
        "--notify-pycontrol",
        action="store_true",
        dest="notify_pycontrol",
        help="Send event marker codes to pyControl over UART.",
    )
    pycontrol.add_argument(
        "--no-notify-pycontrol",
        action="store_false",
        dest="notify_pycontrol",
        help="Do not send event marker codes to pyControl.",
    )
    parser.set_defaults(notify_pycontrol=None)
    parser.add_argument(
        "--pycontrol-port",
        default=None,
        help="Serial port for pyControl event markers. Defaults to config.toml.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after this many scheduled stimulation attempts.",
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
        help="Random seed for balanced shuffling. Generated and logged if omitted.",
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
    if args.max_events is not None and args.max_events <= 0:
        parser.error("--max-events must be greater than 0.")
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


def validate_interval(interval_s):
    try:
        interval_s = float(interval_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("Timed-random protocol requires interval_s.") from exc
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be a finite value greater than 0 seconds.")
    return interval_s


def resolve_protocol(protocol_name=None):
    protocols = CONFIG.get("timed_random_protocols", {})
    selected = protocol_name or CONFIG.get("run", {}).get("active_timed_random_protocol")
    if not selected:
        available = ", ".join(sorted(protocols)) or "(none configured)"
        raise ValueError(
            "No timed-random protocol selected. Set "
            "[run].active_timed_random_protocol or pass --protocol. "
            f"Available protocols: {available}"
        )
    if selected not in protocols:
        available = ", ".join(sorted(protocols)) or "(none configured)"
        raise ValueError(
            f"Unknown timed-random protocol '{selected}'. "
            f"Available protocols: {available}"
        )
    return selected, protocols[selected]


def pycontrol_codes():
    event_conf = CONFIG.get("pycontrol_events", {})
    configured = event_conf.get("codes", {})
    return {**DEFAULT_PYCONTROL_CODES, **configured}


def resolve_pycontrol_port(args, hw_conf, event_conf):
    if args.pycontrol_port:
        return args.pycontrol_port, [args.pycontrol_port], None

    if "ports" in event_conf or "port" in event_conf:
        return resolve_serial_port(
            event_conf,
            "ports",
            "port",
            label="pyControl event port",
            require_available=True,
        )

    return resolve_serial_port(
        hw_conf,
        "trigger_ports",
        "trigger_port",
        label="pyControl event port",
        require_available=True,
    )


def validate_timed_random_protocol(protocol_name, protocol, *, interval_override=None):
    """Validate config and return profile details plus normalized conditions."""
    profiles = CONFIG.get("stimulation", {})
    profile_name = protocol.get("profile")
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none configured)"
        raise ValueError(
            f"Timed-random protocol '{protocol_name}' refers to unknown "
            f"stimulation profile '{profile_name}'. Available profiles: {available}"
        )

    stim_profile = dict(profiles[profile_name])
    pulse_mode = normalize_pulse_mode(stim_profile, validate=True)
    if pulse_mode != "train":
        raise ValueError(
            f"Timed-random protocol '{protocol_name}' requires a train-mode "
            f"profile; '{profile_name}' uses pulse_mode={pulse_mode!r}."
        )

    interval_s = validate_interval(
        interval_override if interval_override is not None else protocol.get("interval_s")
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
            f"Timed-random protocol '{protocol_name}' must define at least one condition."
        )

    conditions = []
    for index, condition in enumerate(raw_conditions, start=1):
        name = str(condition.get("name") or f"condition_{index}")
        if "duration" not in condition:
            raise ValueError(f"Condition '{name}' is missing duration.")
        try:
            duration_s = float(condition["duration"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Condition '{name}' has invalid duration.") from exc
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError(
                f"Condition '{name}' duration must be finite and greater than 0."
            )
        if duration_s >= interval_s:
            raise ValueError(
                f"Condition '{name}' duration {duration_s:g}s must be shorter "
                f"than interval {interval_s:g}s."
            )
        if "amp" not in condition:
            raise ValueError(f"Condition '{name}' is missing amp.")
        amp_values = normalize_stim_values(condition["amp"], channel_count, "amp")
        validate_amp_values(amp_values, max_amp_values, channels)
        sham = bool(condition.get("sham", False)) or all(
            value == 0 for value in amp_values
        )
        conditions.append(
            TimedRandomCondition(
                name=name,
                amp_values=amp_values,
                duration_s=duration_s,
                sham=sham,
                raw=dict(condition),
            )
        )

    return {
        "profile_name": profile_name,
        "stim_profile": stim_profile,
        "pulse_mode": pulse_mode,
        "interval_s": interval_s,
        "channels": channels,
        "pw_values": pw_values,
        "max_amp_values": max_amp_values,
        "conditions": conditions,
    }


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
    status=None,
    amp_values=None,
    pw_values=None,
    **details,
):
    fields = build_event_fields(
        "timed_random",
        profile_name,
        stim_profile,
        stim_port,
        pulse_mode,
        mock_mode,
        experiment=protocol_name,
        duration=duration,
        command=command,
        status=status,
        amp_values=amp_values,
        pw_values=pw_values,
    )
    fields.update(details)
    return fields


def should_stop(event_count, max_events, session_start, runtime_s):
    if max_events is not None and event_count >= max_events:
        return True
    if runtime_s is not None and time.perf_counter() - session_start >= runtime_s:
        return True
    return False


def sleep_until(target_time):
    while True:
        remaining = target_time - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


def main():
    args = parse_args()
    run_conf = CONFIG.get("run", {})

    try:
        protocol_name, protocol = resolve_protocol(args.protocol)
        validated = validate_timed_random_protocol(
            protocol_name, protocol, interval_override=args.interval
        )
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
            f"!! FATAL: Timed-random protocol '{protocol_name}' is not enabled "
            "for real hardware."
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
    except RuntimeError as exc:
        print(f"!! FATAL: {exc}")
        return 2

    profile_name = validated["profile_name"]
    stim_profile = validated["stim_profile"]
    pulse_mode = validated["pulse_mode"]
    interval_s = validated["interval_s"]
    pw_values = validated["pw_values"]
    conditions = validated["conditions"]
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)
    deck = BalancedConditionDeck(len(conditions), rng)

    matlab_path = os.path.join(root_dir, "matlab_backend")
    if not os.path.exists(matlab_path):
        print(f"!! ERROR: MATLAB backend not found at {matlab_path}")
        return 1

    channel_metadata = resolve_channel_metadata(stim_profile)
    log_conf = CONFIG.get("logging", {})
    event_conf = CONFIG.get("pycontrol_events", {})
    notify_pycontrol = bool(event_conf.get("enabled", False))
    if args.notify_pycontrol is not None:
        notify_pycontrol = args.notify_pycontrol
    codes = pycontrol_codes()
    log_enabled = bool(log_conf.get("stim_events_enabled", True)) and not args.no_local_log
    event_logger = make_event_logger(
        script="timed_random_stimulation.py",
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
            "interval_s": interval_s,
            "seed": seed,
            "randomization_mode": RANDOMIZATION_MODE,
            "conditions": [condition.raw for condition in conditions],
            "max_events": args.max_events,
            "runtime_s": args.runtime,
            "pycontrol_events": event_conf,
            "notify_pycontrol": notify_pycontrol,
        },
        enabled=log_enabled,
    )

    pycontrol_link = None

    def send_marker(marker):
        if not marker:
            return None
        code = codes.get(marker)
        if code is None:
            return None
        if pycontrol_link is None:
            return code if notify_pycontrol else None
        try:
            pycontrol_link.send_code(code)
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

    print("=== Timed Random Stimulation ===")
    print(f">> Protocol: {protocol_name}")
    print(f">> Profile: {profile_name}")
    print(
        f">> Schedule: every {interval_s:g}s onset-to-onset, "
        f"{RANDOMIZATION_MODE}, seed={seed}"
    )
    print(
        f">> Protocol settings: {stim_profile['freq']}Hz, pw={pw_values}ms, "
        f"pulse_mode={pulse_mode}"
    )
    print(f">> Mode: {'MOCK' if mock_mode else 'REAL'}")
    print(f">> Stimulator port: {stim_port}")
    if notify_pycontrol:
        print(f">> pyControl marker echo: ON (codes={codes})")
    if len(stim_port_candidates) > 1:
        print(f">> Stimulator port candidates: {stim_port_candidates}")
        if detected_ports is not None:
            print(f">> Detected serial ports: {detected_ports}")
    if not args.quiet:
        print(">> Conditions:")
        for index, condition in enumerate(conditions, start=1):
            label = "sham" if condition.sham else "stim"
            print(
                f"   {index}: {condition.name} ({label}) "
                f"amp={condition.amp_values}mA duration={condition.duration_s:g}s"
            )
    if event_logger.enabled:
        print(f">> Stimulation event log: {event_logger.csv_path}")

    if notify_pycontrol:
        try:
            pycontrol_port, pycontrol_candidates, pycontrol_detected = (
                resolve_pycontrol_port(args, HW_CONF, event_conf)
            )
            pycontrol_link = PyControlEventLink(
                pycontrol_port,
                baud_rate=event_conf.get("baud_rate", HW_CONF.get("baud_rate", 9600)),
            ).open()
        except Exception as exc:
            event_logger.close()
            print(f"!! Could not open pyControl event link: {exc}")
            return 2

        print(f">> pyControl event link: {pycontrol_port}")
        if len(pycontrol_candidates) > 1:
            print(f">> pyControl port candidates: {pycontrol_candidates}")
            if pycontrol_detected is not None:
                print(f">> Detected serial ports: {pycontrol_detected}")

    stim = MatlabStimulator(
        matlab_path,
        mock_mode=mock_mode,
        calibration_dir=CALIBRATION_DIR,
        quiet=args.quiet,
    )
    exit_code = 0
    event_count = 0
    session_start = time.perf_counter()
    next_onset = session_start
    zero_values = [0.0] * len(validated["channels"])

    try:
        session_start_code = send_marker("session_start")
        event_logger.record(
            "session_start",
            **event_fields(
                protocol_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                duration=None,
                amp_values=validated["max_amp_values"],
                pw_values=pw_values,
                pycontrol_code=session_start_code,
                interval_s=interval_s,
                seed=seed,
                randomization_mode=RANDOMIZATION_MODE,
            ),
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
        print(">> Running. Press Ctrl-C to stop.")

        while not should_stop(event_count, args.max_events, session_start, args.runtime):
            if args.runtime is not None and next_onset - session_start >= args.runtime:
                break
            sleep_until(next_onset)
            actual_onset = time.perf_counter()
            lateness_s = max(0.0, actual_onset - next_onset)
            condition_index, shuffle_cycle = deck.next_index()
            condition = conditions[condition_index]
            event_count += 1
            scheduled_onset_s = next_onset - session_start
            actual_onset_s = actual_onset - session_start

            common_details = {
                "condition_index": condition_index + 1,
                "shuffle_cycle": shuffle_cycle,
                "scheduled_onset_s": scheduled_onset_s,
                "actual_onset_s": actual_onset_s,
                "lateness_s": lateness_s,
                "event_number": event_count,
                "interval_s": interval_s,
            }

            if condition.sham:
                sham_code = send_marker("stim_sham")
                if not args.quiet:
                    print(
                        f">> Event {event_count}: {condition.name} sham "
                        f"(cycle {shuffle_cycle})"
                    )
                event_logger.record(
                    "stim_sham",
                    **event_fields(
                        protocol_name,
                        profile_name,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        duration=condition.duration_s,
                        command=condition.name,
                        status="no_stimulation",
                        amp_values=condition.amp_values,
                        pw_values=pw_values,
                        pycontrol_code=sham_code,
                        **common_details,
                    ),
                )
                next_onset += interval_s
                continue

            if not args.quiet:
                print(
                    f">> Event {event_count}: {condition.name} "
                    f"amp={condition.amp_values}mA duration={condition.duration_s:g}s "
                    f"(cycle {shuffle_cycle})"
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
                    duration=condition.duration_s,
                    command=condition.name,
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
                    duration=condition.duration_s,
                    command=condition.name,
                    status="sent",
                    amp_values=condition.amp_values,
                    pw_values=pw_values,
                    pycontrol_code=on_code,
                    **common_details,
                ),
            )
            try:
                time.sleep(condition.duration_s)
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
                        duration=condition.duration_s,
                        command=condition.name,
                        status="sent",
                        amp_values=zero_values,
                        pw_values=pw_values,
                        pycontrol_code=off_code,
                        **common_details,
                    ),
                )
            next_onset += interval_s

    except KeyboardInterrupt:
        print("\n>> Stopping by user request.")
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
        try:
            stim.stop()
        finally:
            stim.close()
            session_end_code = send_marker("session_end")
            event_logger.record(
                "session_end",
                **event_fields(
                    protocol_name,
                    profile_name,
                    stim_profile,
                    stim_port,
                    pulse_mode,
                    mock_mode,
                    duration=None,
                    amp_values=validated["max_amp_values"],
                    pw_values=pw_values,
                    pycontrol_code=session_end_code,
                    events_completed=event_count,
                ),
            )
            if pycontrol_link is not None:
                pycontrol_link.close()
            event_logger.close()
        print("Bye.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
