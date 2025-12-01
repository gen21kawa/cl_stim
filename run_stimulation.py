"""
Main script for closed-loop electrical stimulation
"""

import sys
import os
import time
import struct
import serial

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
from matlab_stimulator import MatlabStimulator

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# Load Hardware Config
STIM_PORT = CONFIG['hardware']['stimulator_port']
TRIGGER_PORT = CONFIG['hardware']['trigger_port']
BAUD_RATE = CONFIG['hardware']['baud_rate']

# Load Stimulation Protocol (e.g., 'brain_standard')
# You can change this key to select different protocols from config.toml
STIM_PROFILE = CONFIG['stimulation']['brain_standard']

USE_MOCK = True # Set to False for real experiment

def main():
    print("=== Stimulation Trigger Server ===")
    print(f">> Protocol: {STIM_PROFILE['freq']}Hz, {STIM_PROFILE['amp']}mA, {STIM_PROFILE['pw']}ms")

    # 1. Initialize Stimulator Driver
    matlab_path = os.path.join(root_dir, 'matlab_backend')
    if not os.path.exists(matlab_path):
        print(f"!! ERROR: MATLAB backend not found at {matlab_path}")
        return

    stim = MatlabStimulator(matlab_path, mock_mode=USE_MOCK)

    # 2. Configure Stimulator
    print(">> Configuring Stimulator...")
    stim.configure(
        port=STIM_PORT,
        freq=STIM_PROFILE['freq'],
        pw=STIM_PROFILE['pw'],
        amp=STIM_PROFILE['amp']
    )
    
    # 3. Connect to Stimulator Hardware
    try:
        stim.connect()
    except Exception as e:
        print(f"!! FATAL: Could not connect to stimulator: {e}")
        return

    # 4. Open Serial Link to Behavior PC
    print(f">> Opening Trigger Link on {TRIGGER_PORT}...")
    try:
        # Timeout is small to allow for non-blocking-ish loop
        link = serial.Serial(TRIGGER_PORT, BAUD_RATE, timeout=0.01)
        print(">> Link Open. Waiting for triggers...")
    except serial.SerialException as e:
        print(f"!! FATAL: Could not open trigger port {TRIGGER_PORT}: {e}")
        stim.close()
        return

    # 5. Main Event Loop
    try:
        while True:
            # PyControl sends 2 bytes (int16, little-endian)
            if link.in_waiting >= 2:
                raw_data = link.read(2)
                try:
                    command_code = struct.unpack('<h', raw_data)[0]
                except struct.error:
                    print("!! Warning: Malformed packet received")
                    continue
                
                # --- TRIGGER LOGIC ---
                if command_code == 1: 
                    print(f">> [Trigger 1] Motion Detected! Stimulating...")
                    
                    # A. Start Stimulation
                    stim.stimulate(pw=STIM_PROFILE['pw'], amp=STIM_PROFILE['amp'])
                    
                    # B. Hold for Duration
                    time.sleep(STIM_PROFILE['duration'])
                    
                    # C. Stop Stimulation
                    stim.stop()
                    print("   Done.")

                elif command_code == 0:
                    print(">> [Trigger 0] Session Start/End signal received.")
                    
                else:
                    print(f">> [Trigger {command_code}] Unknown command ignored.")

            # Yield slightly to CPU
            time.sleep(0.001) 

    except KeyboardInterrupt:
        print("\n>> Server stopping by user request.")
    except Exception as e:
        print(f"\n!! Runtime Error: {e}")
    finally:
        print(">> Shutting down...")
        link.close()
        stim.close()
        print("Bye.")

if __name__ == "__main__":
    main()