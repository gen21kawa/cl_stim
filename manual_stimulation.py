"""
Interactive manual stimulation test script.

This script drives the stimulator from typed user commands instead of
pyControl serial triggers. It uses [run].mock_mode from config.toml by
default; pass --real or --mock to override it.

The main jobs of this script are:
1. Load a stimulation profile from config.toml.
2. Configure and connect the MATLAB/Ripple stimulator wrapper.
3. Let the operator manually send on/off/pulse commands.
4. Optionally notify pyControl that stimulation started/stopped.
5. Always keep a local CSV/JSON audit trail unless disabled.

Timing note:
The local event log records when Python requested/sent commands. For precise
neural/behavior alignment, use hardware TTL/sync when available and treat this
software log as the parameter/audit record.
"""

import argparse
import math
import os
import sys
import time

# Keep imports working when the script is launched directly from this folder.
# The repo root contains matlab_stimulator.py and the utils package.
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


# These defaults are deliberately separate from behavior-trigger command codes.
# pyControl's UARTlink only generates an event when the received integer changes,
# so using distinct session/on/off/pulse values makes repeated manual events
# easier to detect reliably in the pyControl task log.
DEFAULT_PYCONTROL_CODES = {
    "session_start": 110,
    "session_end": 111,
    "stim_on": 101,
    "stim_off": 102,
    "stim_pulse": 103,
}


def parse_args():
    """Parse CLI options that override config.toml for one manual session."""
    parser = argparse.ArgumentParser(
        description="Manually turn stimulation on/off without pyControl triggers."
    )
    parser.add_argument(
        "--profile",
        default="brain_standard",
        help="Stimulation profile name from config.toml (default: brain_standard).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Pulse duration override in seconds for the pulse command.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real stimulator hardware, overriding [run].mock_mode.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock mode, overriding [run].mock_mode.",
    )

    # pyControl notification is optional because manual stim is useful even when
    # the behavior computer/task is not connected. The CLI can override the
    # [pycontrol_events].enabled default for a single run.
    pycontrol = parser.add_mutually_exclusive_group()
    pycontrol.add_argument(
        "--notify-pycontrol",
        action="store_true",
        dest="notify_pycontrol",
        help="Send stimulation event markers to pyControl over UART.",
    )
    pycontrol.add_argument(
        "--no-notify-pycontrol",
        action="store_false",
        dest="notify_pycontrol",
        help="Do not send stimulation event markers to pyControl.",
    )
    parser.set_defaults(notify_pycontrol=None)
    parser.add_argument(
        "--pycontrol-port",
        default=None,
        help="Serial port for pyControl event markers. Defaults to config.toml.",
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
    if args.real and args.mock:
        parser.error("--real and --mock cannot be used together.")
    return args


def print_help():
    """Print the interactive commands accepted at the stim> prompt."""
    print("Commands:")
    print("  on | 1 [mA ...]    Start stimulation at current or given amplitude")
    print("  off | 0            Stop stimulation")
    print("  pulse | 2 [seconds] [mA ...] Stimulate briefly, then stop")
    print("  pulse amp <mA ...> Use default duration at a given amplitude")
    print("  amp [mA ...]       Show/set current amplitude; profile amp is the max")
    print("  status             Show current settings")
    print("  help               Show this command list")
    print("  quit | exit        Stop stimulation and exit")


def profile_pulse_mode(stim_profile):
    """Normalize the profile's pulse mode (lenient; no validation)."""
    return normalize_pulse_mode(stim_profile)


def normalize_stim_values(values, channel_count, name):
    """Return one numeric value per configured stimulation channel."""
    if isinstance(values, (int, float)):
        return [float(values)] * channel_count

    normalized = [float(value) for value in values]
    if len(normalized) == 1:
        return normalized * channel_count
    if len(normalized) != channel_count:
        raise ValueError(
            f"{name} must be a scalar or have one value per configured channel."
        )
    return normalized


def format_stim_values(values):
    """Format scalar-like or per-channel values for operator messages."""
    if len(values) == 1:
        return f"{values[0]:g}"
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def validate_amp_values(amp_values, max_amp_values, channels):
    """Keep manual amplitudes within the configured profile safety ceiling."""
    for channel, value, max_value in zip(channels, amp_values, max_amp_values):
        if not math.isfinite(value) or not math.isfinite(max_value):
            raise ValueError("Amplitude values must be finite numbers.")
        if value < 0:
            raise ValueError("Amplitude must be greater than or equal to 0 mA.")
        if value > max_value + 1e-12:
            raise ValueError(
                f"Amplitude {value:g} mA for channel {channel} exceeds the "
                f"profile maximum of {max_value:g} mA."
            )


def parse_amp_values(tokens, channel_count, max_amp_values, channels):
    """Parse and validate scalar or per-channel amplitude values in mA."""
    if tokens and tokens[0] == "amp":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("Usage: amp <mA> [mA ...]")

    try:
        amp_values = normalize_stim_values(tokens, channel_count, "amp")
    except ValueError as exc:
        raise ValueError(f"Invalid amplitude: {exc}") from exc

    validate_amp_values(amp_values, max_amp_values, channels)
    return amp_values


def parse_on_amp_values(parts, current_amp_values, channel_count, max_amp_values, channels):
    """Return the current or command-specific amplitude for an on command."""
    if len(parts) == 1:
        return list(current_amp_values)
    return parse_amp_values(parts[1:], channel_count, max_amp_values, channels)


def parse_pulse_command(
    parts,
    default_duration,
    current_amp_values,
    channel_count,
    max_amp_values,
    channels,
):
    """Return duration and amplitude for pulse command variants."""
    if len(parts) == 1:
        return default_duration, list(current_amp_values)

    tokens = parts[1:]
    if tokens[0] == "amp":
        amp_values = parse_amp_values(
            tokens[1:], channel_count, max_amp_values, channels
        )
        return default_duration, amp_values

    try:
        duration = float(tokens[0])
    except ValueError as exc:
        raise ValueError("Usage: pulse [seconds] [mA ...]") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Pulse duration must be greater than 0 seconds.")

    if len(tokens) == 1:
        return duration, list(current_amp_values)

    amp_values = parse_amp_values(tokens[1:], channel_count, max_amp_values, channels)
    return duration, amp_values


def print_status(
    profile_name,
    stim_profile,
    stim_port,
    duration,
    mock_mode,
    active,
    current_amp_values,
    max_amp_values,
):
    """Show the current manual-session settings in operator-readable form."""
    mode = "MOCK" if mock_mode else "REAL"
    state = "ON" if active else "OFF"
    pulse_mode = profile_pulse_mode(stim_profile)
    print(f">> Mode: {mode} | State: {state}")
    print(f">> Profile: {profile_name}")
    print(
        f">> Port: {stim_port} | Freq: {stim_profile['freq']}Hz | "
        f"PW: {stim_profile['pw']}ms | "
        f"Amp: {format_stim_values(current_amp_values)}mA "
        f"(profile max {format_stim_values(max_amp_values)}mA) | "
        f"Pulse mode: {pulse_mode} | Pulse duration: {duration}s"
    )


def pycontrol_codes():
    """Merge built-in marker codes with optional config.toml overrides."""
    event_conf = CONFIG.get("pycontrol_events", {})
    configured = event_conf.get("codes", {})
    return {**DEFAULT_PYCONTROL_CODES, **configured}


def resolve_pycontrol_port(args, hw_conf, event_conf):
    """Choose the serial port used for manual-stim markers sent to pyControl."""
    if args.pycontrol_port:
        # Explicit CLI port wins over config and availability checks.
        return args.pycontrol_port, [args.pycontrol_port], None

    if "ports" in event_conf or "port" in event_conf:
        # Preferred modern config location for the pyControl marker link.
        return resolve_serial_port(
            event_conf,
            "ports",
            "port",
            label="pyControl event port",
            require_available=True,
        )

    # Backward-compatible fallback: reuse the existing hardware trigger_port.
    return resolve_serial_port(
        hw_conf,
        "trigger_ports",
        "trigger_port",
        label="pyControl event port",
        require_available=True,
    )


def event_fields(
    profile_name,
    stim_profile,
    stim_port,
    pulse_mode,
    mock_mode,
    duration=None,
    command=None,
    status=None,
    amp_values=None,
    pw_values=None,
):
    """Manual-mode adapter around the shared :func:`build_event_fields`."""
    return build_event_fields(
        "manual",
        profile_name,
        stim_profile,
        stim_port,
        pulse_mode,
        mock_mode,
        duration=duration,
        command=command,
        status=status,
        amp_values=amp_values,
        pw_values=pw_values,
    )


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load and validate the selected stimulation profile.
    # ------------------------------------------------------------------
    hw_conf = CONFIG["hardware"]
    stim_profiles = CONFIG["stimulation"]
    if args.profile not in stim_profiles:
        available = ", ".join(sorted(stim_profiles))
        print(f"!! Unknown stimulation profile: {args.profile}")
        print(f"!! Available profiles: {available}")
        return 2

    stim_profile = dict(stim_profiles[args.profile])
    pulse_mode = profile_pulse_mode(stim_profile)
    channels = MatlabStimulator._normalize_channels(stim_profile.get("channel"))
    channel_count = len(channels)
    pw_values = normalize_stim_values(stim_profile["pw"], channel_count, "pw")
    if any(not math.isfinite(value) or value < 0 for value in pw_values):
        print("!! Invalid profile pulse width: values must be finite and >= 0 ms.")
        return 2
    max_amp_values = normalize_stim_values(stim_profile["amp"], channel_count, "amp")
    try:
        validate_amp_values(max_amp_values, max_amp_values, channels)
    except ValueError as exc:
        print(f"!! Invalid profile amplitude: {exc}")
        return 2
    current_amp_values = list(max_amp_values)
    zero_values = [0.0] * channel_count

    # "train" means stimulation stays active until we send zeros/stop.
    # "single_pulse" means each command triggers a short run-once train.
    if pulse_mode not in ("train", "single_pulse"):
        print("!! pulse_mode must be 'train' or 'single_pulse'.")
        return 2
    pulse_duration = (
        args.duration if args.duration is not None else stim_profile.get("duration", 0.5)
    )
    if pulse_mode != "single_pulse" and (
        not math.isfinite(pulse_duration) or pulse_duration <= 0
    ):
        print("!! Pulse duration must be a finite value greater than 0 seconds.")
        return 2

    matlab_path = os.path.join(root_dir, "matlab_backend")
    if not os.path.exists(matlab_path):
        print(f"!! ERROR: MATLAB backend not found at {matlab_path}")
        return 1

    # Config chooses the normal mode; CLI flags are intentional per-run
    # overrides so you can dry-run safely or force real hardware explicitly.
    mock_mode = bool(CONFIG.get("run", {}).get("mock_mode", True))
    if args.real:
        mock_mode = False
    elif args.mock:
        mock_mode = True

    try:
        stim_port, stim_port_candidates, detected_ports = resolve_serial_port(
            hw_conf,
            "stimulator_ports",
            "stimulator_port",
            label="stimulator port",
            require_available=not mock_mode,
        )
    except RuntimeError as exc:
        print(f"!! {exc}")
        return 2

    # ------------------------------------------------------------------
    # Prepare local session logging and optional pyControl notification.
    # ------------------------------------------------------------------
    calibration_dir = hw_conf.get("calibration_dir")
    log_conf = CONFIG.get("logging", {})
    event_conf = CONFIG.get("pycontrol_events", {})
    channel_metadata = resolve_channel_metadata(stim_profile)

    # Local logging is on by default because it is the stim/ephys computer's
    # durable record of what parameters were requested and when.
    log_enabled = bool(log_conf.get("stim_events_enabled", True)) and not args.no_local_log
    event_logger = make_event_logger(
        script="manual_stimulation.py",
        animal=args.animal,
        session_id=args.session_id,
        data_root=args.data_root,
        log_dir=args.log_dir,
        root_dir=root_dir,
        profile_name=args.profile,
        stim_profile=stim_profile,
        channel_metadata=channel_metadata,
        stim_port=stim_port,
        mock_mode=mock_mode,
        extra={"pycontrol_events": event_conf},
        enabled=log_enabled,
    )

    # pyControl notification is intentionally separate from local logging:
    # the UART cable/task may not be running during bench tests.
    notify_pycontrol = bool(event_conf.get("enabled", False))
    if args.notify_pycontrol is not None:
        notify_pycontrol = args.notify_pycontrol

    pycontrol_link = None
    codes = pycontrol_codes()

    def record_event(event, marker=None, **fields):
        """Write one local event row and, when requested, send a pyControl code.

        The local CSV is always attempted first/last around the same event, but
        the pyControl UART marker is only sent for events with a named marker
        such as "stim_on" or "stim_off".
        """
        code = codes.get(marker) if marker else None
        if marker and pycontrol_link is not None:
            try:
                pycontrol_link.send_code(code)
                fields.setdefault("status", "sent")
            except Exception as exc:
                event_logger.record(
                    event,
                    pycontrol_code=code,
                    error=str(exc),
                    status="pycontrol_send_failed",
                    **fields,
                )
                raise
        elif marker:
            # Keep the intended marker code in the local CSV even when no
            # pyControl link is open. This makes post hoc comparison easier.
            fields.setdefault("status", "local_only")

        event_logger.record(event, pycontrol_code=code, **fields)

    print("=== Manual Stimulation Test ===")
    print_status(
        args.profile,
        stim_profile,
        stim_port,
        pulse_duration,
        mock_mode,
        active=False,
        current_amp_values=current_amp_values,
        max_amp_values=max_amp_values,
    )
    if mock_mode:
        print(">> Mock mode is active. Pass --real to use stimulator hardware.")
    else:
        print(">> Real mode is active. Pass --mock to force a no-hardware run.")
    if len(stim_port_candidates) > 1:
        print(f">> Stimulator port candidates: {stim_port_candidates}")
        if detected_ports is not None:
            print(f">> Detected serial ports: {detected_ports}")
    if event_logger.enabled:
        print(f">> Stimulation event log: {event_logger.csv_path}")

    if notify_pycontrol:
        try:
            # The host writes the same 2-byte little-endian integer format read
            # by TreadmillTasks/devices/UARTlink.py on the pyControl side.
            pycontrol_port, pycontrol_candidates, pycontrol_detected = resolve_pycontrol_port(
                args, hw_conf, event_conf
            )
            pycontrol_link = PyControlEventLink(
                pycontrol_port,
                baud_rate=event_conf.get("baud_rate", hw_conf.get("baud_rate", 9600)),
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
    else:
        print(">> pyControl event link disabled. Pass --notify-pycontrol to enable it.")

    stim = MatlabStimulator(
        matlab_path, mock_mode=mock_mode, calibration_dir=calibration_dir
    )
    active = False

    try:
        # Record session start before hardware configuration so failed setup is
        # still associated with a session_id and profile in the local log.
        record_event(
            "session_start",
            marker="session_start",
            **event_fields(
                args.profile,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                duration=pulse_duration,
                amp_values=current_amp_values,
                pw_values=pw_values,
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

        # connect() starts/initializes the MATLAB/Ripple hardware unless mock
        # mode is active. In mock mode it just marks the wrapper as ready.
        print(">> Connecting...")
        stim.connect()
        print_help()

        while True:
            # This is the operator control loop. The commands are deliberately
            # tiny and memorable because this may be used during an experiment.
            command = input("stim> ").strip().lower()
            if not command:
                continue

            parts = command.split()
            action = parts[0]

            try:
                if action in ("on", "1"):
                    # Avoid repeated "on" commands because they are ambiguous
                    # for event alignment and can hide a missing "off".
                    if active:
                        print(">> Stimulation already ON")
                        record_event(
                            "stim_on_ignored",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=pulse_duration,
                                command=command,
                                status="already_active",
                                amp_values=current_amp_values,
                                pw_values=pw_values,
                            ),
                        )
                        continue

                    stim_amp_values = parse_on_amp_values(
                        parts,
                        current_amp_values,
                        channel_count,
                        max_amp_values,
                        channels,
                    )
                    current_amp_values = list(stim_amp_values)
                    record_event(
                        "stim_on_request",
                        **event_fields(
                            args.profile,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            duration=pulse_duration,
                            command=command,
                            status="requested",
                            amp_values=stim_amp_values,
                            pw_values=pw_values,
                        ),
                    )
                    stim.stimulate(pw=pw_values, amp=stim_amp_values)

                    # In train mode, nonzero PW means the stimulator should stay
                    # active until stop/zero PW. In single-pulse mode, the
                    # command is over immediately after the run-once trigger.
                    active = pulse_mode != "single_pulse"
                    if pulse_mode == "single_pulse":
                        expected_duration = stim.single_pulse_train_ms / 1000
                        record_event(
                            "stim_pulse",
                            marker="stim_pulse",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=expected_duration,
                                command=command,
                                amp_values=stim_amp_values,
                                pw_values=pw_values,
                            ),
                        )
                        print(
                            ">> Single pulse sent "
                            f"at {format_stim_values(stim_amp_values)}mA"
                        )
                    else:
                        record_event(
                            "stim_on",
                            marker="stim_on",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=pulse_duration,
                                command=command,
                                amp_values=stim_amp_values,
                                pw_values=pw_values,
                            ),
                        )
                        print(
                            ">> Stimulation ON "
                            f"at {format_stim_values(stim_amp_values)}mA"
                        )

                elif action in ("off", "0"):
                    # For train mode, an off command is only meaningful if an on
                    # command is active. Single-pulse mode is already idle.
                    if not active and pulse_mode != "single_pulse":
                        print(">> Stimulation already OFF")
                        record_event(
                            "stim_off_ignored",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=pulse_duration,
                                command=command,
                                status="already_inactive",
                                amp_values=zero_values,
                                pw_values=zero_values,
                            ),
                        )
                        continue

                    record_event(
                        "stim_off_request",
                        **event_fields(
                            args.profile,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            duration=pulse_duration,
                            command=command,
                            status="requested",
                            amp_values=zero_values,
                            pw_values=zero_values,
                        ),
                    )
                    stim.stop()
                    active = False

                    # stim_off is logged after stim.stop() returns, so this row
                    # approximates "Python finished sending the stop command".
                    record_event(
                        "stim_off",
                        marker="stim_off",
                        **event_fields(
                            args.profile,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            duration=pulse_duration,
                            command=command,
                            amp_values=zero_values,
                            pw_values=zero_values,
                        ),
                    )
                    print(">> Stimulation OFF")

                elif action in ("pulse", "2"):
                    if active:
                        print(">> Turn stimulation off before sending a pulse")
                        record_event(
                            "stim_pulse_ignored",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=pulse_duration,
                                command=command,
                                status="already_active",
                                amp_values=current_amp_values,
                                pw_values=pw_values,
                            ),
                        )
                        continue

                    duration, stim_amp_values = parse_pulse_command(
                        parts,
                        pulse_duration,
                        current_amp_values,
                        channel_count,
                        max_amp_values,
                        channels,
                    )
                    current_amp_values = list(stim_amp_values)

                    # A pulse command is a complete on -> wait -> off cycle in
                    # train mode. In single-pulse mode, it is just one run-once
                    # trigger and no explicit wait is needed here.
                    record_event(
                        "stim_pulse_request",
                        **event_fields(
                            args.profile,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            duration=duration,
                            command=command,
                            status="requested",
                            amp_values=stim_amp_values,
                            pw_values=pw_values,
                        ),
                    )
                    if pulse_mode == "single_pulse":
                        print(
                            ">> Single pulse "
                            f"at {format_stim_values(stim_amp_values)}mA"
                        )
                        stim.stimulate(pw=pw_values, amp=stim_amp_values)
                        active = False
                        record_event(
                            "stim_pulse",
                            marker="stim_pulse",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=stim.single_pulse_train_ms / 1000,
                                command=command,
                                amp_values=stim_amp_values,
                                pw_values=pw_values,
                            ),
                        )
                        print(">> Pulse complete")
                        continue

                    print(
                        f">> Pulse ON for {duration}s "
                        f"at {format_stim_values(stim_amp_values)}mA"
                    )
                    stim.stimulate(pw=pw_values, amp=stim_amp_values)
                    active = True

                    # Send/log stim_on only after the stimulation command is
                    # issued, so pyControl gets the marker near the Python-side
                    # command time rather than before we try to stimulate.
                    record_event(
                        "stim_on",
                        marker="stim_on",
                        **event_fields(
                            args.profile,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            duration=duration,
                            command=command,
                            amp_values=stim_amp_values,
                            pw_values=pw_values,
                        ),
                    )
                    try:
                        time.sleep(duration)
                    finally:
                        # Use finally so a KeyboardInterrupt or runtime error
                        # during the wait still attempts to stop stimulation.
                        stim.stop()
                        active = False
                        record_event(
                            "stim_off",
                            marker="stim_off",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=duration,
                                command=command,
                                amp_values=zero_values,
                                pw_values=zero_values,
                            ),
                        )
                    print(">> Pulse complete")

                elif action == "amp":
                    if len(parts) == 1:
                        print(
                            ">> Current amp: "
                            f"{format_stim_values(current_amp_values)}mA "
                            f"(profile max {format_stim_values(max_amp_values)}mA)"
                        )
                        continue

                    if active:
                        print(">> Turn stimulation off before changing amplitude")
                        record_event(
                            "amp_change_ignored",
                            **event_fields(
                                args.profile,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                duration=pulse_duration,
                                command=command,
                                status="already_active",
                                amp_values=current_amp_values,
                                pw_values=pw_values,
                            ),
                        )
                        continue

                    current_amp_values = parse_amp_values(
                        parts[1:],
                        channel_count,
                        max_amp_values,
                        channels,
                    )
                    record_event(
                        "amp_change",
                        **event_fields(
                            args.profile,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            duration=pulse_duration,
                            command=command,
                            status="updated",
                            amp_values=current_amp_values,
                            pw_values=pw_values,
                        ),
                    )
                    print(
                        ">> Current amp set to "
                        f"{format_stim_values(current_amp_values)}mA"
                    )

                elif action == "status":
                    print_status(
                        args.profile,
                        stim_profile,
                        stim_port,
                        pulse_duration,
                        mock_mode,
                        active,
                        current_amp_values,
                        max_amp_values,
                    )

                elif action == "help":
                    print_help()

                elif action in ("quit", "exit"):
                    break

                else:
                    # Unknown commands are not logged as stimulation events
                    # because they do not correspond to a hardware attempt.
                    print(f"!! Unknown command: {action}")
                    print("!! Type 'help' for valid commands.")

            except ValueError as exc:
                # Handles bad pulse duration syntax while keeping the manual
                # session alive.
                print(f"!! {exc}")

    except KeyboardInterrupt:
        print("\n>> Stopping by user request.")
    except Exception as exc:
        print(f"\n!! Runtime Error: {exc}")
        return 1
    finally:
        print(">> Shutting down...")
        try:
            # If the session exits while train stimulation is active, force a
            # stop and log it. This is the most important safety path here.
            if active:
                record_event(
                    "stim_off_request",
                    **event_fields(
                        args.profile,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        duration=pulse_duration,
                        command="shutdown",
                        status="requested",
                        amp_values=zero_values,
                        pw_values=zero_values,
                    ),
                )
            stim.stop()
            if active:
                record_event(
                    "stim_off",
                    marker="stim_off",
                    **event_fields(
                        args.profile,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        duration=pulse_duration,
                        command="shutdown",
                        amp_values=zero_values,
                        pw_values=zero_values,
                    ),
                )
        finally:
            # Always close MATLAB/stimulator resources before closing the event
            # log and pyControl UART link. close() also sends a hardware stop
            # command inside MatlabStimulator.
            stim.close()
            try:
                record_event(
                    "session_end",
                    marker="session_end",
                    **event_fields(
                        args.profile,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        duration=pulse_duration,
                        amp_values=current_amp_values,
                        pw_values=pw_values,
                    ),
                )
            finally:
                if pycontrol_link is not None:
                    pycontrol_link.close()
                event_logger.close()
        print("Bye.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
