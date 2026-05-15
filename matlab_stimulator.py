import matlab.engine
import os
import sys

class MatlabStimulator:
    """
    A Python wrapper for the MATLAB Stimulation Engine.
    Handles the initialization of the MATLAB runtime, setting up the 'sp' structure,
    and driving the wireless stimulator hardware.
    """

    def __init__(self, matlab_backend_path, mock_mode=False):
        """
        Initialize the MATLAB Engine and add the backend files to the path.
        
        Args:
            matlab_backend_path (str): Absolute path to the folder containing .m files.
            mock_mode (bool): If True, bypasses actual hardware calls for testing.
        """
        self.mock = mock_mode
        self.ws_initialized = False
        
        print("[MatlabStimulator] Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()
        
        # Verify path exists
        if not os.path.exists(matlab_backend_path):
            raise FileNotFoundError(f"MATLAB backend path not found: {matlab_backend_path}")
            
        # Add the backend folder to MATLAB's search path
        print(f"[MatlabStimulator] Adding path: {matlab_backend_path}")
        self.eng.addpath(self.eng.genpath(matlab_backend_path))

    def configure(self, port, freq=30, pw=0.2, amp=1.0, return_mode='monopolar'):
        """
        Sets up the 'sp' (stimulation parameters) structure in the MATLAB workspace.
        This replaces the old 'bmi_params_defaults.m'.
        """
        print(f"[MatlabStimulator] Configuring: Port={port}, Freq={freq}Hz, Mode={return_mode}")
        
        # 1. Create the struct
        self.eng.eval("sp = struct();", nargout=0)
        self.eng.eval("sp.output = 'wireless_stim';", nargout=0)
        self.eng.eval("sp.mode = 'PW_modulation';", nargout=0)
        self.eng.eval(f"sp.return = '{return_mode}';", nargout=0)
        
        # 2. Hardware Settings
        self.eng.eval(f"sp.serial_string = '{port}';", nargout=0)
        self.eng.eval(f"sp.freq = {freq};", nargout=0)
        self.eng.eval("sp.inter_ph_int = 50e-6;", nargout=0)
        
        # 3. Define Muscles & Maps (Hardcoded to 1 channel for triggers usually)
        # You can make this dynamic if you need multi-channel control
        self.eng.eval("sp.muscles = {'Channel1'};", nargout=0)
        
        # Anode Map: {[1]; [1]} -> Channel 1, 100% current
        # MATLAB syntax: {{[1]; [1]}}
        self.eng.eval("sp.anode_map = {{[1]; [1]}};", nargout=0)
        
        # Cathode Map: Empty for monopolar
        self.eng.eval("sp.cathode_map = {{}};", nargout=0)
        
        # 4. Safety Limits (Critical)
        # We set the 'max' values to the config values so the mapper scales 1.0 -> max
        self.eng.eval(f"sp.PW_max = {pw} * ones(1, 1);", nargout=0)
        self.eng.eval(f"sp.amplitude_max = {amp} * ones(1, 1);", nargout=0)
        
        # 5. Mapping Logic (0-1 input)
        self.eng.eval("sp.EMG_min = 0; sp.EMG_max = 1;", nargout=0)
        self.eng.eval("sp.PW_min = 0; sp.amplitude_min = 0;", nargout=0)
        self.eng.eval("sp.stim_resolut = 0.001;", nargout=0)

    def connect(self):
        """
        Initializes the Wireless Stimulator hardware object.
        Calls 'setup_wireless_stim_fes' to push config.
        """
        if self.mock:
            print("[MatlabStimulator] MOCK: Hardware connected successfully.")
            self.ws_initialized = True
            return

        try:
            print("[MatlabStimulator] Connecting to dongle...")
            # Create object with debug level 1
            self.eng.eval("stim_params = struct('dbg_lvl', 1, 'serial_string', sp.serial_string);", nargout=0)
            self.eng.eval("ws = wireless_stim(stim_params);", nargout=0)
            
            # Initialize & Version Check
            self.eng.eval("ws.init();", nargout=0)
            # self.eng.eval("ws.version();", nargout=0) # Optional
            
            # Send static configuration (Frequency, Polarity)
            self.eng.eval("setup_wireless_stim_fes(ws, sp);", nargout=0)
            
            self.ws_initialized = True
            print("[MatlabStimulator] Hardware Armed & Ready.")
            
        except Exception as e:
            print(f"[MatlabStimulator] CONNECTION ERROR: {e}")
            raise ConnectionError("Failed to initialize stimulator hardware.")

    def stimulate(self, pw, amp):
        """
        Sends a command to the stimulator.
        
        Args:
            pw (float): Pulse width in milliseconds (e.g., 0.2)
            amp (float): Amplitude in mA (e.g., 0.05)
        """
        if self.mock:
            print(f"[MatlabStimulator] MOCK FIRE: {pw}ms @ {amp}mA")
            return

        if not self.ws_initialized:
            print("[MatlabStimulator] Ignored: Hardware not initialized.")
            return

        try:
            # 1. Update MATLAB variables
            self.eng.eval(f"current_PW = {pw};", nargout=0)
            self.eng.eval(f"current_Amp = {amp};", nargout=0)
            
            # 2. Run Mapping Function
            # Returns 'cmds' struct and 'ch_list'
            self.eng.eval("[cmds, ch_list] = stim_elect_mapping_wireless(current_PW, current_Amp, sp);", nargout=0)
            
            # 3. Send Command via Object
            self.eng.eval("for k = 1:numel(cmds), ws.set_stim(cmds{k}, ch_list); end", nargout=0)
            
        except Exception as e:
            print(f"[MatlabStimulator] STIM ERROR: {e}")

    def stop(self):
        """Safely stops stimulation (0 PW)"""
        self.stimulate(0, 0)

    def close(self):
        """Clean shutdown of hardware and engine"""
        print("[MatlabStimulator] Shutting down...")
        if not self.mock and self.ws_initialized:
            try:
                # Stop command specific to wireless stimulator
                self.eng.eval("ws.set_Run(ws.run_stop, 1:ws.num_channels);", nargout=0)
                self.eng.eval("delete(ws);", nargout=0)
            except:
                pass
        
        self.eng.quit()
        print("[MatlabStimulator] Engine Closed.")
