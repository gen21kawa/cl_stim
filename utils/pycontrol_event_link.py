"""Host-side pyControl event marker link.

The matching pyControl task can use the UARTlink device from TreadmillTasks,
which reads 2-byte little-endian integers and exposes the latest value as
``hw.bci_link.spk``.
"""


class PyControlEventLink:
    """Send integer event markers to a pyControl UART link."""

    def __init__(self, port, baud_rate=9600, timeout=0.01):
        self.port = port
        self.baud_rate = int(baud_rate)
        self.timeout = timeout
        self.serial = None

    def open(self):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed in this Python environment. "
                "Use the conda environment declared in env.yml or install pyserial."
            ) from exc

        self.serial = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        return self

    def send_code(self, code):
        """Send one unsigned 16-bit little-endian marker code."""
        if self.serial is None:
            raise RuntimeError("pyControl event link is not open.")
        if code < 0 or code > 65535:
            raise ValueError("pyControl event codes must fit in 2 unsigned bytes.")

        self.serial.write(int(code).to_bytes(2, byteorder="little", signed=False))
        self.serial.flush()

    def close(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None
