"General Logger"
# Copied from real-time-neuropixels (implemented by Mostafa Safaie)

import logging
import os
import time
import multiprocessing
import sys
import shutil
import traceback
from pathlib import Path


class StreamToLogger:
    """
    Redirects writes to a logger, replacing '\n' with '|'.
    """

    def __init__(self, logger, log_level=logging.INFO, delimiter="|"):
        self.logger = logger
        self.log_level = log_level
        self.delimiter = delimiter

    def write(self, buf):
        """
        Replace all newline characters in `buf` with the chosen delimiter.
        Then log each line at the specified level.
        """
        msg = str(buf).replace(" ", "").replace("\n", self.delimiter)
        if len(msg) > 3:  # Ignore empty lines
            self.logger.log(self.log_level, msg)

    def flush(self):
        """
        Required method for file-like objects; no action needed here.
        """
        pass


class CpuTimeFilter(logging.Filter):
    def filter(self, record):
        # Adds the CPU time in seconds to the LogRecord
        try:
            record.cpu_time = (
                time.perf_counter()
            )  # time.perf_counter() is system-wide since python 3.8
        except Exception as e:
            # If for some reason perf_counter fails, set cpu_time to -1
            # and log an error to stderr (or handle as needed)
            sys.stderr.write(f"Error getting CPU time: {str(e)}\n")
            record.cpu_time = -1
        return True


def get_logger() -> logging.Logger:
    """
    Sets up the main logger to send all logs.
    """
    logger_name = multiprocessing.current_process().name

    # Rename the Main process to Process-Main
    if logger_name == "MainProcess":
        logger_name = "Process-Main"
    # If already configured, return the existing logger
    logger_obj = logging.getLogger(logger_name)

    # Configure the logger only if it has no handlers yet
    if not logger_obj.handlers:
        # Set up the main logger to debug level
        logger_obj.setLevel(logging.DEBUG)
        # Add the CPU time filter so that each record includes cpu_time before being queued
        logger_obj.addFilter(CpuTimeFilter())

        formatter = logging.Formatter(
            "%(processName)s: %(asctime)s_%(cpu_time).6f - %(levelname)s: %(message)s",
            datefmt="%Y_%m_%d_%H_%M_%S",
        )

        file_handler = logging.FileHandler(f"{logger_name}.log", mode="w")
        console_handler = logging.StreamHandler()

        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger_obj.addHandler(file_handler)
        logger_obj.addHandler(console_handler)

        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            joined_traceback = "|".join(line.replace("\n", "|") for line in lines)
            logger_obj.error(f"Uncaught exception: {joined_traceback}")

        # # Set the global exception hook for the main process
        if logger_name == "Process-Main":
            sys.excepthook = handle_exception
        else:
            sys.stderr = StreamToLogger(logger_obj, logging.ERROR, delimiter="|")

        logger_obj.debug(f"{logger_name} logger initialised.")
        logger_obj.debug(f"process PID: {multiprocessing.current_process().pid}")

    return logger_obj


def copy_log_files(session_path: Path):
    """
    copy the log files to the session directory `session_path`.
    """
    logging.shutdown()
    # Copy the log file to the desired location
    for log_file in Path(".").glob("*.log"):
        shutil.copy(log_file, Path(str(session_path) + '_' + str(log_file)))
    # If needed, remove the original log file
    # os.remove('BCI-session.log')

def del_log_files():
    """
    Deletes the log files in the repo.
    """
    for log_file in Path(".").glob("*.log"):
        if 'Main' not in log_file.name and 'Process' in log_file.name:
            os.remove(log_file)
