import sys
import os
import time
import math

# Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__)) 
print(current_dir)
root_dir = os.path.dirname(current_dir)
print(root_dir)
sys.path.append(root_dir)

# Imports
from matlab_stimulator import MatlabStimulator
from utils.loader import CONFIG  # <--- Import the loader

# CONFIGURATION
USE_MOCK = True
DURATION = 5.0

def main():
    print("=== Stimulation Hardware Test (TOML Config) ===")
    
    # 1. Load Settings from TOML
    hw_conf = CONFIG['hardware']
    stim_conf = CONFIG['stimulation']['brain_standard'] # Pick a profile
    
    print(f">> Loaded Profile: Brain Standard")
    print(f"   Port: {hw_conf['stimulator_port']}")
    print(f"   Freq: {stim_conf['freq']} Hz | Amp: {stim_conf['amp']} mA")

    # 2. Initialize Driver
    matlab_path = os.path.join(root_dir, 'matlab_backend')
    if not os.path.exists(matlab_path):
        print(f"!! ERROR: MATLAB path not found: {matlab_path}")
        return

    stim = MatlabStimulator(matlab_path, mock_mode=USE_MOCK)

    # 3. Configure Hardware
    stim.configure(
        port=hw_conf['stimulator_port'],
        freq=stim_conf['freq'],
        pw=stim_conf['pw'],
        amp=stim_conf['amp']
    )

    # 4. Connect
    stim.connect()

    # 5. Run Test Pattern
    print(f">> Running Test Pattern for {DURATION} seconds...")
    start_time = time.time()

    try:
        while (time.time() - start_time) < DURATION:
            t = time.time() - start_time
            intensity = (math.sin(t * 4) + 1) / 2
            
            test_pw = intensity * stim_conf['pw']
            test_amp = stim_conf['amp']
            
            stim.stimulate(pw=test_pw, amp=test_amp)
            
            if USE_MOCK:
                print(f"   [MOCK] PW: {test_pw:.4f} ms | Amp: {test_amp} mA")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n>> Aborted.")
    finally:
        stim.close()
        print("Done.")

if __name__ == "__main__":
    main()