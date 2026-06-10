import os


class MatlabStimulator:
    """
    A Python wrapper for the MATLAB Stimulation Engine.
    Handles the initialization of the MATLAB runtime, setting up the 'sp' structure,
    and driving the wireless stimulator hardware.
    """

    def __init__(self, matlab_backend_path, mock_mode=False, calibration_dir=None):
        """
        Initialize the MATLAB Engine and add the backend files to the path.
        
        Args:
            matlab_backend_path (str): Absolute path to the folder containing .m files.
            mock_mode (bool): If True, bypasses actual hardware calls for testing.
        """
        self.mock = mock_mode
        self.calibration_dir = calibration_dir
        self.ws_initialized = False
        self.eng = None
        self.channels = (1,)
        self.inter_phase = 50e-6
        self.pulse_mode = "train"
        self.single_pulse_train_ms = 1.0
        
        # Verify path exists
        if not os.path.exists(matlab_backend_path):
            raise FileNotFoundError(f"MATLAB backend path not found: {matlab_backend_path}")

        if self.mock:
            print("[MatlabStimulator] MOCK: MATLAB Engine not started.")
            return

        import matlab.engine

        print("[MatlabStimulator] Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()
            
        # Add the backend folder to MATLAB's search path
        print(f"[MatlabStimulator] Adding path: {matlab_backend_path}")
        self.eng.addpath(self.eng.genpath(matlab_backend_path))

    @staticmethod
    def _normalize_channels(channels):
        if channels is None:
            channels = [1]
        elif isinstance(channels, int):
            channels = [channels]

        normalized = tuple(int(channel) for channel in channels)
        if not normalized:
            raise ValueError("At least one stimulation channel is required.")
        if any(channel < 1 or channel > 16 for channel in normalized):
            raise ValueError("Stimulation channels must be between 1 and 16.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Stimulation channels must be unique.")
        return normalized

    @staticmethod
    def _matlab_row(values):
        return "[" + " ".join(str(float(value)) for value in values) + "]"

    @staticmethod
    def _normalize_pulse_mode(pulse_mode):
        normalized = str(pulse_mode).lower()
        if normalized == "single":
            normalized = "single_pulse"
        if normalized not in ("train", "single_pulse"):
            raise ValueError("pulse_mode must be 'train' or 'single_pulse'.")
        return normalized

    @staticmethod
    def _default_single_pulse_train_ms(pw_values, inter_phase):
        max_pw_ms = max(float(value) for value in pw_values)
        inter_phase_ms = float(inter_phase) * 1000
        return max(1.0, 0.05 + (2 * max_pw_ms) + inter_phase_ms + 0.1)

    def _normalize_stim_values(self, values, name):
        if isinstance(values, (int, float)):
            return [float(values)] * len(self.channels)

        normalized = [float(value) for value in values]
        if len(normalized) == 1:
            return normalized * len(self.channels)
        if len(normalized) != len(self.channels):
            raise ValueError(
                f"{name} must be a scalar or have one value per configured channel."
            )
        return normalized

    def _send_stim_values(self, pw_values, amp_values):
        self.eng.eval(f"current_PW = {self._matlab_row(pw_values)};", nargout=0)
        self.eng.eval(f"current_Amp = {self._matlab_row(amp_values)};", nargout=0)
        self.eng.eval(
            "[cmds, ch_list] = stim_elect_mapping_wireless(current_PW, current_Amp, sp);",
            nargout=0,
        )
        self.eng.eval(
            "for k = 1:numel(cmds), ws.set_stim(cmds{k}, ch_list); end",
            nargout=0,
        )

    def configure(
        self,
        port,
        freq=30,
        pw=0.2,
        amp=1.0,
        return_mode='monopolar',
        channels=None,
        inter_phase=50e-6,
        pulse_mode="train",
        single_pulse_train_ms=None,
    ):
        """
        Sets up the 'sp' (stimulation parameters) structure in the MATLAB workspace.
        This replaces the old 'bmi_params_defaults.m'.
        """
        self.channels = self._normalize_channels(channels)
        self.inter_phase = float(inter_phase)
        self.pulse_mode = self._normalize_pulse_mode(pulse_mode)
        pw_values = self._normalize_stim_values(pw, "pw")
        amp_values = self._normalize_stim_values(amp, "amp")
        if single_pulse_train_ms is None:
            single_pulse_train_ms = self._default_single_pulse_train_ms(
                pw_values, self.inter_phase
            )
        self.single_pulse_train_ms = float(single_pulse_train_ms)
        if self.single_pulse_train_ms <= 0:
            raise ValueError("single_pulse_train_ms must be greater than 0.")

        channel_count = len(self.channels)
        channel_row = self._matlab_row(self.channels)
        pw_row = self._matlab_row(pw_values)
        amp_row = self._matlab_row(amp_values)
        muscle_names = ", ".join(f"'Channel{channel}'" for channel in self.channels)
        anode_map = (
            "{"
            + ", ".join(str(channel) for channel in self.channels)
            + "; "
            + ", ".join("1" for _ in self.channels)
            + "}"
        )

        print(
            f"[MatlabStimulator] Configuring: Port={port}, Freq={freq}Hz, "
            f"Mode={return_mode}, PulseMode={self.pulse_mode}, "
            f"Channels={list(self.channels)}"
        )

        if self.mock:
            print(
                f"[MatlabStimulator] MOCK CONFIG: PW={pw_values}ms, "
                f"Amp={amp_values}mA, Inter-phase={self.inter_phase}s, "
                f"SinglePulseTrain={self.single_pulse_train_ms}ms"
            )
            return
        
        # 1. Create the struct
        self.eng.eval("sp = struct();", nargout=0)
        self.eng.eval("sp.output = 'wireless_stim';", nargout=0)
        self.eng.eval("sp.mode = 'PW_modulation';", nargout=0)
        self.eng.eval(f"sp.return = '{return_mode}';", nargout=0)
        
        # 2. Hardware Settings
        self.eng.eval(f"sp.serial_string = '{port}';", nargout=0)
        self.eng.eval(f"sp.freq = {freq};", nargout=0)
        self.eng.eval(f"sp.inter_ph_int = {self.inter_phase};", nargout=0)
        self.eng.eval(f"sp.pulse_mode = '{self.pulse_mode}';", nargout=0)
        self.eng.eval(
            f"sp.single_pulse_train_ms = {self.single_pulse_train_ms};",
            nargout=0,
        )
        
        # 3. Define Muscles & Maps from the selected stimulation profile.
        self.eng.eval(f"sp.muscles = {{{muscle_names}}};", nargout=0)
        
        # Anode Map row 1: channels; row 2: current fraction per channel.
        self.eng.eval(f"sp.anode_map = {anode_map};", nargout=0)
        
        # Cathode Map: Empty for monopolar
        self.eng.eval("sp.cathode_map = {{}};", nargout=0)
        
        # 4. Safety Limits (Critical)
        # We set the 'max' values to the config values so the mapper scales 1.0 -> max
        self.eng.eval(f"sp.PW_max = {pw_row};", nargout=0)
        self.eng.eval(f"sp.amplitude_max = {amp_row};", nargout=0)
        
        # 5. Mapping Logic (0-1 input)
        self.eng.eval(
            f"sp.EMG_min = zeros(1, {channel_count}); "
            f"sp.EMG_max = ones(1, {channel_count});",
            nargout=0,
        )
        self.eng.eval(
            f"sp.PW_min = zeros(1, {channel_count}); "
            f"sp.amplitude_min = zeros(1, {channel_count});",
            nargout=0,
        )
        self.eng.eval(f"sp.channel = {channel_row};", nargout=0)
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
            self.eng.eval("stim_params = struct('dbg_lvl', 1, 'comm_timeout_ms', -1, 'blocking', true, 'zb_ch_page', 17, 'serial_string', sp.serial_string, 'trim_calibrate_if_missing', false);", nargout=0)
            self.eng.eval("ws = wireless_stim(stim_params);", nargout=0)
            
            # Initialize & Version Check
            self.eng.eval("previous_dir = pwd;", nargout=0)
            try:
                if self.calibration_dir:
                    self.eng.cd(self.calibration_dir, nargout=0)
                self.eng.eval("ws.init();", nargout=0)
            finally:
                self.eng.eval("cd(previous_dir);", nargout=0)
            self.eng.eval("ws.version();", nargout=0)
            
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
            pw_values = self._normalize_stim_values(pw, "pw")
            amp_values = self._normalize_stim_values(amp, "amp")
            label = "MOCK SINGLE PULSE" if self.pulse_mode == "single_pulse" else "MOCK FIRE"
            print(
                f"[MatlabStimulator] {label}: channels={list(self.channels)}, "
                f"pw={pw_values}ms, amp={amp_values}mA"
            )
            return

        if not self.ws_initialized:
            print("[MatlabStimulator] Ignored: Hardware not initialized.")
            return

        try:
            pw_values = self._normalize_stim_values(pw, "pw")
            amp_values = self._normalize_stim_values(amp, "amp")
            self._send_stim_values(pw_values, amp_values)

            if self.pulse_mode == "single_pulse" and any(value > 0 for value in pw_values):
                active_channels = [
                    channel for channel, value in zip(self.channels, pw_values) if value > 0
                ]
                active_channel_row = self._matlab_row(active_channels)
                self.eng.eval(
                    f"ws.set_Run(ws.run_once, {active_channel_row});",
                    nargout=0,
                )
                self.eng.eval(
                    f"ws.set_Run(ws.run_once_go, {active_channel_row});",
                    nargout=0,
                )
            
        except Exception as e:
            print(f"[MatlabStimulator] STIM ERROR: {e}")

    def stop(self):
        """Safely stops stimulation (0 PW)"""
        if self.mock:
            if self.pulse_mode == "single_pulse":
                print("[MatlabStimulator] MOCK STOP: single-pulse mode idle.")
                return
            self.stimulate(0, 0)
            return

        if self.pulse_mode == "single_pulse":
            try:
                zeros = [0.0] * len(self.channels)
                self._send_stim_values(zeros, zeros)
                if self.ws_initialized:
                    self.eng.eval(
                        "ws.set_Run(ws.run_stop, 1:ws.num_channels);",
                        nargout=0,
                    )
            except Exception as e:
                print(f"[MatlabStimulator] STOP ERROR: {e}")
            return

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

        if self.eng is not None:
            self.eng.quit()
        print("[MatlabStimulator] Engine Closed.")
