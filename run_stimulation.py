"""
Main script for closed-loop electrical stimulation
"""

import argparse
import os
import struct
import sys
import time

# -------------------------------------------------------------------------
# SETUP PATHS
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir # Since this script is in the root
sys.path.append(root_dir)

# -------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------
from utils.loader import CONFIG
from utils.serial_port_resolver import resolve_serial_port
from utils.stim_event_log import (
    StimEventLogger,
    resolve_channel_map,
    resolve_output_dir,
    resolve_session_dir,
)
from matlab_stimulator import MatlabStimulator

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# Load Hardware Config
HW_CONF = CONFIG['hardware']
TRIGGER_PORT = CONFIG['hardware']['trigger_port']
BAUD_RATE = CONFIG['hardware']['baud_rate']
CALIBRATION_DIR = CONFIG['hardware'].get('calibration_dir')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the UART-triggered stimulation server."
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment name from config.toml. Defaults to [run].active_experiment.",
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
        help="Use real stimulator hardware. Requires experiment real_enabled = true.",
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
    return parser.parse_args()


def expand_to_channels(value, channel_count, name):
    if isinstance(value, (int, float)):
        return [float(value)] * channel_count

    values = [float(item) for item in value]
    if len(values) == 1:
        return values * channel_count
    if len(values) != channel_count:
        raise ValueError(
            f"{name} must be a scalar or have one value per configured channel."
        )
    return values


def load_experiment(experiment_name):
    experiments = CONFIG.get("experiments", {})
    if experiment_name not in experiments:
        available = ", ".join(sorted(experiments)) or "(none configured)"
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. Available experiments: {available}"
        )

    experiment = experiments[experiment_name]
    profile_name = experiment.get("profile")
    profiles = CONFIG.get("stimulation", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none configured)"
        raise ValueError(
            f"Experiment '{experiment_name}' refers to unknown stimulation profile "
            f"'{profile_name}'. Available profiles: {available}"
        )

    commands = {}
    for raw_code, command in experiment.get("commands", {}).items():
        try:
            code = int(raw_code)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Experiment '{experiment_name}' has a non-integer command code: "
                f"{raw_code}"
            ) from exc
        if code == 0:
            raise ValueError(
                f"Experiment '{experiment_name}' cannot use command 0; "
                "0 is reserved for session start/end markers."
            )
        commands[code] = command

    if not commands:
        raise ValueError(f"Experiment '{experiment_name}' has no command definitions.")

    return experiment, profile_name, profiles[profile_name], commands


def command_pw_values(command, base_pw_values, channel_count):
    fractions = command.get("pw_fraction")
    if fractions is None:
        raise ValueError(f"Command '{command.get('name', '(unnamed)')}' missing pw_fraction.")

    pw_fraction = expand_to_channels(fractions, channel_count, "pw_fraction")
    if any(fraction < 0 or fraction > 1 for fraction in pw_fraction):
        raise ValueError("pw_fraction values must be between 0 and 1.")

    return [base_pw * fraction for base_pw, fraction in zip(base_pw_values, pw_fraction)]


def pulse_mode_from_profile(stim_profile):
    pulse_mode = str(stim_profile.get("pulse_mode", "train")).lower()
    if pulse_mode == "single":
        pulse_mode = "single_pulse"
    if pulse_mode not in ("train", "single_pulse"):
        raise ValueError("pulse_mode must be 'train' or 'single_pulse'.")
    return pulse_mode


def validate_commands(commands, base_pw_values, channel_count, default_duration, pulse_mode):
    if pulse_mode == "train" and default_duration <= 0:
        raise ValueError("Train-mode stimulation profiles require duration > 0.")

    for code, command in commands.items():
        command_pw_values(command, base_pw_values, channel_count)
        if pulse_mode == "train":
            duration = float(command.get("duration", default_duration))
            if duration <= 0:
                raise ValueError(f"Command {code} has non-positive duration {duration}.")


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def open_trigger_link(port, baud_rate):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed in this Python environment. "
            "Use the conda environment declared in env.yml or install pyserial."
        ) from exc

    return serial.Serial(port, baud_rate, timeout=0.01)


def event_fields(
    experiment_name,
    profile_name,
    stim_profile,
    stim_port,
    pulse_mode,
    mock_mode,
    amp_values=None,
    pw_values=None,
    duration=None,
    command=None,
    command_code=None,
    status=None,
):
    channel_metadata = stim_profile.get("_channel_metadata", {})
    fields = {
        "source": "pycontrol_trigger_server",
        "experiment": experiment_name,
        "profile": profile_name,
        "pulse_mode": pulse_mode,
        "channels": stim_profile.get("channel", [1]),
        "channel_labels": channel_metadata.get("channel_labels"),
        "physical_contacts": channel_metadata.get("physical_contacts"),
        "freq_hz": stim_profile.get("freq"),
        "pw_ms": pw_values if pw_values is not None else stim_profile.get("pw"),
        "amp_mA": amp_values if amp_values is not None else stim_profile.get("amp"),
        "duration_s": duration,
        "inter_phase_s": stim_profile.get("inter_phase"),
        "stim_port": stim_port,
        "mock_mode": mock_mode,
        "command": command,
        "command_code": command_code,
        "status": status,
    }
    return {key: value for key, value in fields.items() if value is not None}

def main():
    args = parse_args()
    run_conf = CONFIG.get("run", {})
    experiment_name = args.experiment or run_conf.get("active_experiment")
    if not experiment_name:
        print("!! FATAL: No experiment selected. Set [run].active_experiment or pass --experiment.")
        return 2

    try:
        experiment, profile_name, stim_profile, commands = load_experiment(experiment_name)
        stim_profile = dict(stim_profile)
    except ValueError as exc:
        print(f"!! FATAL: {exc}")
        return 2

    mock_mode = bool(run_conf.get("mock_mode", True))
    if args.real:
        mock_mode = False
    elif args.mock:
        mock_mode = True

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

    if not mock_mode and not experiment.get("real_enabled", False):
        print(
            f"!! FATAL: Experiment '{experiment_name}' is not enabled for real hardware."
        )
        print(
            "!! Set real_enabled = true only after replacing placeholder channels "
            "and bench-checking the setup."
        )
        return 2

    log_triggers = bool(run_conf.get("log_triggers", True))
    channels = stim_profile.get("channel")
    channel_count = len(MatlabStimulator._normalize_channels(channels))
    try:
        pulse_mode = pulse_mode_from_profile(stim_profile)
        default_duration = float(stim_profile.get("duration", 0))
        base_pw_values = expand_to_channels(stim_profile["pw"], channel_count, "pw")
        amp_values = expand_to_channels(stim_profile["amp"], channel_count, "amp")
        validate_commands(
            commands,
            base_pw_values,
            channel_count,
            default_duration,
            pulse_mode,
        )
    except ValueError as exc:
        print(f"!! FATAL: {exc}")
        return 2

    print("=== Stimulation Trigger Server ===")
    print(f">> Experiment: {experiment_name}")
    print(f">> Profile: {profile_name}")
    print(
        f">> Protocol: {stim_profile['freq']}Hz, amp={amp_values}mA, "
        f"pw={base_pw_values}ms, pulse_mode={pulse_mode}"
    )
    if pulse_mode == "train":
        print(f">> Train duration: {default_duration}s")
    else:
        print(
            f">> Single-pulse train length: "
            f"{stim_profile.get('single_pulse_train_ms', 'auto')}ms"
        )
    print(f">> Mode: {'MOCK' if mock_mode else 'REAL'}")
    print(f">> Stimulator port: {stim_port}")
    if len(stim_port_candidates) > 1:
        print(f">> Stimulator port candidates: {stim_port_candidates}")
        if detected_ports is not None:
            print(f">> Detected serial ports: {detected_ports}")
    print(">> Command map:")
    for code in sorted(commands):
        command = commands[code]
        print(f"   {code}: {command.get('name', '(unnamed)')} pw_fraction={command.get('pw_fraction')}")

    # 1. Initialize Stimulator Driver
    matlab_path = os.path.join(root_dir, 'matlab_backend')
    if not os.path.exists(matlab_path):
        print(f"!! ERROR: MATLAB backend not found at {matlab_path}")
        return 1

    log_conf = CONFIG.get("logging", {})
    channel_metadata = resolve_channel_map(
        stim_profile.get("channel"), CONFIG.get("channel_map", {})
    )
    stim_profile["_channel_metadata"] = channel_metadata
    for warning in channel_metadata["warnings"]:
        print(f"!! Channel map warning: {warning}")

    log_enabled = bool(log_conf.get("stim_events_enabled", True)) and not args.no_local_log
    session_dir, resolved_session_id = resolve_session_dir(
        args.animal,
        args.session_id,
        args.data_root or log_conf.get("data_root", "data"),
        root_dir,
    )
    log_dir = resolve_output_dir(
        args.log_dir or log_conf.get("stim_event_dir", "logs/stim_events"),
        root_dir,
    )
    event_logger = StimEventLogger(
        log_dir,
        session_id=resolved_session_id,
        session_dir=session_dir,
        metadata={
            "script": "run_stimulation.py",
            "animal": args.animal,
            "session_dir": session_dir,
            "experiment_name": experiment_name,
            "profile_name": profile_name,
            "stim_profile": stim_profile,
            "channel_map": channel_metadata["channel_map"],
            "channel_map_warnings": channel_metadata["warnings"],
            "commands": commands,
            "stim_port": stim_port,
            "trigger_port": TRIGGER_PORT,
            "mock_mode": mock_mode,
        },
        enabled=log_enabled,
    )
    if event_logger.enabled:
        print(f">> Stimulation event log: {event_logger.csv_path}")
    event_logger.record(
        "session_start",
        **event_fields(
            experiment_name,
            profile_name,
            stim_profile,
            stim_port,
            pulse_mode,
            mock_mode,
            amp_values=amp_values,
            pw_values=base_pw_values,
            duration=default_duration if pulse_mode == "train" else None,
        ),
    )

    stim = MatlabStimulator(matlab_path, mock_mode=mock_mode, calibration_dir=CALIBRATION_DIR)

    # 2. Configure Stimulator
    print(">> Configuring Stimulator...")
    stim.configure(
        port=stim_port,
        freq=stim_profile['freq'],
        pw=stim_profile['pw'],
        amp=stim_profile['amp'],
        channels=channels,
        inter_phase=stim_profile.get('inter_phase', 50e-6),
        pulse_mode=pulse_mode,
        single_pulse_train_ms=stim_profile.get("single_pulse_train_ms"),
    )
    
    # 3. Connect to Stimulator Hardware
    try:
        stim.connect()
    except Exception as e:
        print(f"!! FATAL: Could not connect to stimulator: {e}")
        event_logger.record(
            "stimulator_connect_failed",
            error=str(e),
            **event_fields(
                experiment_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                status="failed",
            ),
        )
        event_logger.close()
        stim.close()
        return 1

    # 4. Open Serial Link to Behavior PC
    print(f">> Opening Trigger Link on {TRIGGER_PORT}...")
    link = None
    try:
        # Timeout is small to allow for non-blocking-ish loop
        link = open_trigger_link(TRIGGER_PORT, BAUD_RATE)
        print(">> Link Open. Waiting for triggers...")
    except Exception as e:
        print(f"!! FATAL: Could not open trigger port {TRIGGER_PORT}: {e}")
        event_logger.record(
            "trigger_link_open_failed",
            error=str(e),
            **event_fields(
                experiment_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                status="failed",
                command=TRIGGER_PORT,
            ),
        )
        event_logger.close()
        stim.close()
        return 1

    # 5. Main Event Loop
    exit_code = 0
    try:
        while True:
            # PyControl sends 2 bytes (int16, little-endian)
            if link.in_waiting >= 2:
                raw_data = link.read(2)
                try:
                    command_code = struct.unpack('<h', raw_data)[0]
                except struct.error:
                    print("!! Warning: Malformed packet received")
                    event_logger.record(
                        "trigger_malformed",
                        **event_fields(
                            experiment_name,
                            profile_name,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            status="ignored",
                        ),
                    )
                    continue
                event_logger.record(
                    "trigger_received",
                    **event_fields(
                        experiment_name,
                        profile_name,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        amp_values=amp_values,
                        pw_values=base_pw_values,
                        command_code=command_code,
                        status="received",
                    ),
                )
                
                # --- TRIGGER LOGIC ---
                if command_code == 0:
                    if log_triggers:
                        print(f">> {timestamp()} [Trigger 0] Session start/end marker.")
                    event_logger.record(
                        "session_marker_received",
                        **event_fields(
                            experiment_name,
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

                if command_code not in commands:
                    print(f">> [Trigger {command_code}] Unknown command ignored.")
                    event_logger.record(
                        "trigger_unknown",
                        **event_fields(
                            experiment_name,
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

                command = commands[command_code]
                try:
                    pw_values = command_pw_values(command, base_pw_values, channel_count)
                except ValueError as exc:
                    print(f"!! [Trigger {command_code}] Invalid command config: {exc}")
                    event_logger.record(
                        "trigger_invalid",
                        error=str(exc),
                        **event_fields(
                            experiment_name,
                            profile_name,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            command=command.get("name", f"command_{command_code}"),
                            command_code=command_code,
                            status="ignored",
                        ),
                    )
                    continue

                command_name = command.get("name", f"command_{command_code}")
                duration = None
                if pulse_mode == "train":
                    duration = float(command.get("duration", default_duration))
                    if duration <= 0:
                        print(
                            f"!! [Trigger {command_code}] Invalid duration "
                            f"{duration}; ignored."
                        )
                        event_logger.record(
                            "trigger_invalid_duration",
                            **event_fields(
                                experiment_name,
                                profile_name,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                amp_values=amp_values,
                                pw_values=pw_values,
                                duration=duration,
                                command=command_name,
                                command_code=command_code,
                                status="ignored",
                            ),
                        )
                        continue

                if log_triggers:
                    print(
                        f">> {timestamp()} [Trigger {command_code}] "
                        f"{command_name}: pw={pw_values}ms amp={amp_values}mA "
                        f"pulse_mode={pulse_mode}"
                    )
                    if pulse_mode == "train":
                        print(f"   duration={duration}s")

                if all(pw == 0 for pw in pw_values):
                    if log_triggers:
                        print("   Sham command: no stimulation sent.")
                    event_logger.record(
                        "stim_sham",
                        **event_fields(
                            experiment_name,
                            profile_name,
                            stim_profile,
                            stim_port,
                            pulse_mode,
                            mock_mode,
                            amp_values=amp_values,
                            pw_values=pw_values,
                            duration=duration,
                            command=command_name,
                            command_code=command_code,
                            status="no_stimulation",
                        ),
                    )
                    continue

                event_logger.record(
                    "stim_on_request",
                    **event_fields(
                        experiment_name,
                        profile_name,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        amp_values=amp_values,
                        pw_values=pw_values,
                        duration=duration,
                        command=command_name,
                        command_code=command_code,
                        status="requested",
                    ),
                )
                stim.stimulate(pw=pw_values, amp=amp_values)
                event_logger.record(
                    "stim_pulse" if pulse_mode == "single_pulse" else "stim_on",
                    **event_fields(
                        experiment_name,
                        profile_name,
                        stim_profile,
                        stim_port,
                        pulse_mode,
                        mock_mode,
                        amp_values=amp_values,
                        pw_values=pw_values,
                        duration=(
                            stim.single_pulse_train_ms / 1000
                            if pulse_mode == "single_pulse"
                            else duration
                        ),
                        command=command_name,
                        command_code=command_code,
                        status="sent",
                    ),
                )
                if pulse_mode == "train":
                    try:
                        time.sleep(duration)
                    finally:
                        stim.stop()
                        event_logger.record(
                            "stim_off",
                            **event_fields(
                                experiment_name,
                                profile_name,
                                stim_profile,
                                stim_port,
                                pulse_mode,
                                mock_mode,
                                amp_values=amp_values,
                                pw_values=pw_values,
                                duration=duration,
                                command=command_name,
                                command_code=command_code,
                                status="sent",
                            ),
                        )
                if log_triggers:
                    print("   Done.")

            # Yield slightly to CPU
            time.sleep(0.001) 

    except KeyboardInterrupt:
        print("\n>> Server stopping by user request.")
    except Exception as e:
        print(f"\n!! Runtime Error: {e}")
        exit_code = 1
    finally:
        print(">> Shutting down...")
        if link is not None:
            link.close()
        stim.close()
        event_logger.record(
            "session_end",
            **event_fields(
                experiment_name,
                profile_name,
                stim_profile,
                stim_port,
                pulse_mode,
                mock_mode,
                amp_values=amp_values,
                pw_values=base_pw_values,
                duration=default_duration if pulse_mode == "train" else None,
            ),
        )
        event_logger.close()
        print("Bye.")
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
