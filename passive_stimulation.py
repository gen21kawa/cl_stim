"""Passive UART stimulation server.

This wrapper exists so the experimental command reads like the intended role:
pyControl owns timing and this process waits for stimulation commands. All
implementation lives in run_stimulation.py.
"""

from run_stimulation import main


if __name__ == "__main__":
    raise SystemExit(main())
