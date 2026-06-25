"""Extract annotated stimulation-aligned video clips."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stim_video_clips import prepare_animal_clip_jobs, run_clip_jobs


DEFAULT_ANIMAL_ROOT = PROJECT_ROOT / "data" / "raw" / "M114"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract annotated video clips around matched stimulation windows."
    )
    parser.add_argument(
        "animal_root",
        nargs="?",
        default=DEFAULT_ANIMAL_ROOT,
        type=Path,
        help=f"Animal raw-data folder. Defaults to {DEFAULT_ANIMAL_ROOT}.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=None,
        help="Stimulation-motion summary folder. Defaults to analysis/stim_motion/<animal_id>.",
    )
    parser.add_argument(
        "--camera-name",
        default="Camera_4",
        help="Camera name to extract from metadata and video files.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Session ID to process. May be passed more than once.",
    )
    parser.add_argument(
        "--pre-s",
        type=float,
        required=True,
        help="Seconds to include before stimulation onset.",
    )
    parser.add_argument(
        "--post-s",
        type=float,
        required=True,
        help="Seconds to include after stimulation offset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ffmpeg commands without writing clips.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clips.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable to use.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_root = (
        args.summary_root
        if args.summary_root is not None
        else PROJECT_ROOT / "analysis" / "stim_motion" / args.animal_root.name
    )

    if args.pre_s < 0 or args.post_s < 0:
        print("--pre-s and --post-s must be non-negative.", file=sys.stderr)
        return 2

    if not args.dry_run and shutil.which(args.ffmpeg) is None:
        print(f"Could not find ffmpeg executable: {args.ffmpeg}", file=sys.stderr)
        return 2

    jobs = prepare_animal_clip_jobs(
        args.animal_root,
        summary_root,
        camera_name=args.camera_name,
        pre_s=args.pre_s,
        post_s=args.post_s,
        sessions=args.session,
        ffmpeg=args.ffmpeg,
    )
    if not jobs:
        print("No matched stimulation video clips found to extract.")
        return 0

    result = run_clip_jobs(jobs, dry_run=args.dry_run, overwrite=args.overwrite)
    action = "prepared" if args.dry_run else "written"
    print(
        f"{result.prepared} clips {action}; "
        f"{result.written} written; {result.skipped} skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
