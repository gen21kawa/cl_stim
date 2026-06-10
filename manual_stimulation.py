"""
Interactive manual stimulation test script.

This script drives the stimulator from typed user commands instead of
pyControl serial triggers. It uses [run].mock_mode from config.toml by
default; pass --real or --mock to override it.
"""

import argparse
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir
sys.path.append(root_dir)

from matlab_stimulator import MatlabStimulator
from utils.loader import CONFIG
from utils.serial_port_resolver import resolve_serial_port


def parse_args():
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
    args = parser.parse_args()
    if args.real and args.mock:
        parser.error("--real and --mock cannot be used together.")
    return args


def print_help():
    print("Commands:")
    print("  on | 1             Start stimulation")
    print("  off | 0            Stop stimulation")
    print("  pulse | 2 [seconds] Stimulate briefly, then stop")
    print("  status             Show current settings")
    print("  help               Show this command list")
    print("  quit | exit        Stop stimulation and exit")


def profile_pulse_mode(stim_profile):
    pulse_mode = str(stim_profile.get("pulse_mode", "train")).lower()
    if pulse_mode == "single":
        pulse_mode = "single_pulse"
    return pulse_mode


def print_status(profile_name, stim_profile, stim_port, duration, mock_mode, active):
    mode = "MOCK" if mock_mode else "REAL"
    state = "ON" if active else "OFF"
    pulse_mode = profile_pulse_mode(stim_profile)
    print(f">> Mode: {mode} | State: {state}")
    print(f">> Profile: {profile_name}")
    print(
        f">> Port: {stim_port} | Freq: {stim_profile['freq']}Hz | "
        f"PW: {stim_profile['pw']}ms | Amp: {stim_profile['amp']}mA | "
        f"Pulse mode: {pulse_mode} | Pulse duration: {duration}s"
    )


def parse_pulse_duration(parts, default_duration):
    if len(parts) == 1:
        return default_duration
    if len(parts) != 2:
        raise ValueError("Usage: pulse [seconds]")

    duration = float(parts[1])
    if duration <= 0:
        raise ValueError("Pulse duration must be greater than 0 seconds.")
    return duration


def main():
    args = parse_args()

    hw_conf = CONFIG["hardware"]
    stim_profiles = CONFIG["stimulation"]
    if args.profile not in stim_profiles:
        available = ", ".join(sorted(stim_profiles))
        print(f"!! Unknown stimulation profile: {args.profile}")
        print(f"!! Available profiles: {available}")
        return 2

    stim_profile = stim_profiles[args.profile]
    pulse_mode = profile_pulse_mode(stim_profile)
    if pulse_mode not in ("train", "single_pulse"):
        print("!! pulse_mode must be 'train' or 'single_pulse'.")
        return 2
    pulse_duration = (
        args.duration if args.duration is not None else stim_profile.get("duration", 0.5)
    )
    if pulse_mode != "single_pulse" and pulse_duration <= 0:
        print("!! Pulse duration must be greater than 0 seconds.")
        return 2

    matlab_path = os.path.join(root_dir, "matlab_backend")
    if not os.path.exists(matlab_path):
        print(f"!! ERROR: MATLAB backend not found at {matlab_path}")
        return 1

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

    calibration_dir = hw_conf.get("calibration_dir")

    print("=== Manual Stimulation Test ===")
    print_status(
        args.profile,
        stim_profile,
        stim_port,
        pulse_duration,
        mock_mode,
        active=False,
    )
    if mock_mode:
        print(">> Mock mode is active. Pass --real to use stimulator hardware.")
    else:
        print(">> Real mode is active. Pass --mock to force a no-hardware run.")
    if len(stim_port_candidates) > 1:
        print(f">> Stimulator port candidates: {stim_port_candidates}")
        if detected_ports is not None:
            print(f">> Detected serial ports: {detected_ports}")

    stim = MatlabStimulator(
        matlab_path, mock_mode=mock_mode, calibration_dir=calibration_dir
    )
    active = False

    try:
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
        print_help()

        while True:
            command = input("stim> ").strip().lower()
            if not command:
                continue

            parts = command.split()
            action = parts[0]

            try:
                if action in ("on", "1"):
                    stim.stimulate(pw=stim_profile["pw"], amp=stim_profile["amp"])
                    active = pulse_mode != "single_pulse"
                    if pulse_mode == "single_pulse":
                        print(">> Single pulse sent")
                    else:
                        print(">> Stimulation ON")

                elif action in ("off", "0"):
                    stim.stop()
                    active = False
                    print(">> Stimulation OFF")

                elif action in ("pulse", "2"):
                    duration = parse_pulse_duration(parts, pulse_duration)
                    if pulse_mode == "single_pulse":
                        print(">> Single pulse")
                        stim.stimulate(pw=stim_profile["pw"], amp=stim_profile["amp"])
                        active = False
                        print(">> Pulse complete")
                        continue

                    print(f">> Pulse ON for {duration}s")
                    stim.stimulate(pw=stim_profile["pw"], amp=stim_profile["amp"])
                    active = True
                    try:
                        time.sleep(duration)
                    finally:
                        stim.stop()
                        active = False
                    print(">> Pulse complete")

                elif action == "status":
                    print_status(
                        args.profile,
                        stim_profile,
                        stim_port,
                        pulse_duration,
                        mock_mode,
                        active,
                    )

                elif action == "help":
                    print_help()

                elif action in ("quit", "exit"):
                    break

                else:
                    print(f"!! Unknown command: {action}")
                    print("!! Type 'help' for valid commands.")

            except ValueError as exc:
                print(f"!! {exc}")

    except KeyboardInterrupt:
        print("\n>> Stopping by user request.")
    except Exception as exc:
        print(f"\n!! Runtime Error: {exc}")
        return 1
    finally:
        print(">> Shutting down...")
        try:
            stim.stop()
        finally:
            stim.close()
        print("Bye.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
