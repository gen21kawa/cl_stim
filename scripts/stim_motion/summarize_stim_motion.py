"""Build stimulation motion summary CSVs for one animal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stim_motion_summary import summarize_animal


DEFAULT_ANIMAL_ROOT = PROJECT_ROOT / "data" / "raw" / "M114"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MotSen1 motion during each stimulation attempt."
    )
    parser.add_argument(
        "animal_root",
        nargs="?",
        default=DEFAULT_ANIMAL_ROOT,
        type=Path,
        help=f"Animal raw-data folder. Defaults to {DEFAULT_ANIMAL_ROOT}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output folder. Defaults to analysis/stim_motion/<animal_id>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, written_paths = summarize_animal(args.animal_root, output_root=args.output_root)
    matched = sum(row.get("matched_pycontrol_window") == "True" for row in rows)
    print(f"Wrote {len(rows)} stimulation rows ({matched} matched pyControl windows).")
    for path in written_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
