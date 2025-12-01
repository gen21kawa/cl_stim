"wrapper for communicating with pycontrol, creates serial port connection and sends int"
# modified from real-time-neuropixels (implemented by Mostafa Safaie)
import serial

from detect_com import serial_ports
from ..main_logger import get_logger

logger = get_logger()


class Controller:
    def __init__(self, byte_per_int=2, baudrate=None):
        """
        class responsible for interfacing with PYC.

        byte_per_int : the number of bytes for every message sent, should match with how pycontrol decodes the message
        baudrate : baudrate default is 9600, should match with pycontrol's setting on decoding the message
        """
        ports = serial_ports()  # detect a list of available com ports
        if len(ports) < 1:
            logger.critical("No serial ports detected")
            raise FileNotFoundError("No serial ports detected")

        self.port = ports[0]  # the first port should be 'COM4' on windows
        self.ser = serial.Serial(self.port)  # create serial communication
        if baudrate:
            self.ser.baudrate = baudrate  # by default baudrate is 9600
        self.byte_per_int = byte_per_int
        self.prev_freq = 0
        logger.debug(f"serial ports: {self.ser}")

    def __str__(self):
        return str(self.ser)

    def send_int(self, freq: int):
        """send integer to pycontrol if it is different from last bin"""
        self.ser.write(freq.to_bytes(self.byte_per_int, byteorder="little"))
        if freq != self.prev_freq:
            self.prev_freq = freq
