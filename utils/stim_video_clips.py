"""Extract annotated stimulation-aligned video clips."""

from __future__ import annotations

import bisect
import csv
import math
import re
import shlex
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/Library/Fonts/Arial.ttf"),
)


@dataclass(frozen=True)
class CameraInputs:
    """Camera files needed for one session."""

    session_dir: Path
    video_path: Path
    metadata_path: Path
    camera_name: str


@dataclass(frozen=True)
class CameraFrame:
    """One camera frame row from metadata."""

    video_index: int
    frame_id: int
    timestamp_ns: int
    relative_ms: float


@dataclass(frozen=True)
class FrameWindow:
    """Video frame range and annotation timing for one stimulation clip."""

    start_index: int
    end_index_exclusive: int
    clip_start_ms: float
    clip_end_ms: float
    stim_on_output_s: float
    stim_off_output_s: float
    fps: float

    @property
    def frame_count(self) -> int:
        return self.end_index_exclusive - self.start_index


@dataclass(frozen=True)
class ClipJob:
    """A prepared stimulation-video extraction job."""

    session_id: str
    event_number: str
    amplitude_folder: str
    input_video: Path
    output_path: Path
    frame_window: FrameWindow
    ffmpeg_command: list[str]
    summary_row: dict[str, str]


@dataclass(frozen=True)
class RunResult:
    """Counts returned after running or dry-running clip jobs."""

    prepared: int
    written: int
    skipped: int


def _as_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bool_cell(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_numeric_tag(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}".replace(".", "p")


def seconds_tag(value: float) -> str:
    """Return a compact path-safe seconds tag."""

    return f"{_format_numeric_tag(value)}s"


def sanitize_path_component(value: object, *, fallback: str = "unknown") -> str:
    """Return a simple filesystem-safe path component."""

    text = str(value).strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or fallback


def amplitude_folder_name(amplitude_uA: object, is_sham: object = False) -> str:
    """Return a normalized amplitude folder name."""

    amplitude = _as_float(amplitude_uA)
    if not math.isfinite(amplitude):
        name = "unknown_uA"
    elif abs(amplitude - round(amplitude)) < 1e-9:
        name = f"{int(round(amplitude)):03d}_uA"
    else:
        name = f"{_format_numeric_tag(amplitude)}_uA"

    if _bool_cell(is_sham):
        name = f"{name}_sham"
    return name


def discover_camera_inputs(
    session_dir: str | Path,
    *,
    camera_name: str = "Camera_4",
) -> CameraInputs:
    """Find one camera video and matching metadata CSV for a session."""

    session_path = Path(session_dir)
    if not session_path.is_dir():
        raise FileNotFoundError(f"Session folder does not exist: {session_path}")

    camera_dirs = sorted(path for path in session_path.glob("*_cameras") if path.is_dir())
    video_candidates: list[Path] = []
    for camera_dir in camera_dirs:
        video_candidates.extend(
            sorted(
                path
                for path in camera_dir.iterdir()
                if path.is_file()
                and path.stem == camera_name
                and path.suffix.lower() in VIDEO_SUFFIXES
            )
        )

    if not video_candidates:
        video_candidates = sorted(
            path
            for path in session_path.rglob(f"{camera_name}.*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    if not video_candidates:
        raise FileNotFoundError(f"Could not find {camera_name} video in {session_path}")

    metadata_candidates = [
        *(camera_dir / "metadata.csv" for camera_dir in camera_dirs),
        session_path / "metadata.csv",
    ]
    metadata_candidates = [path for path in metadata_candidates if path.exists()]
    if not metadata_candidates:
        metadata_candidates = sorted(session_path.rglob("metadata.csv"))

    metadata_path = None
    for candidate in metadata_candidates:
        if metadata_contains_camera(candidate, camera_name):
            metadata_path = candidate
            break
    if metadata_path is None:
        raise FileNotFoundError(
            f"Could not find metadata.csv with {camera_name} rows in {session_path}"
        )

    return CameraInputs(
        session_dir=session_path,
        video_path=video_candidates[0],
        metadata_path=metadata_path,
        camera_name=camera_name,
    )


def metadata_contains_camera(metadata_path: str | Path, camera_name: str) -> bool:
    """Return whether a metadata CSV has at least one row for ``camera_name``."""

    with Path(metadata_path).open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if row.get("frame_camera_name") == camera_name:
                return True
    return False


def load_camera_timeline(
    metadata_path: str | Path,
    *,
    camera_name: str = "Camera_4",
) -> list[CameraFrame]:
    """Load the frame timeline for one camera from camera metadata."""

    frames: list[tuple[int, int]] = []
    with Path(metadata_path).open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {"frame_camera_name", "frame_id", "frame_timestamp"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata missing columns {sorted(missing)}: {metadata_path}")

        for row in reader:
            if row.get("frame_camera_name") != camera_name:
                continue
            try:
                frame_id = int(row["frame_id"])
                timestamp_ns = int(row["frame_timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame metadata row in {metadata_path}: {row}") from exc
            frames.append((frame_id, timestamp_ns))

    if not frames:
        raise ValueError(f"No {camera_name} rows in metadata: {metadata_path}")

    first_timestamp = frames[0][1]
    return [
        CameraFrame(
            video_index=index,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            relative_ms=(timestamp_ns - first_timestamp) / 1_000_000.0,
        )
        for index, (frame_id, timestamp_ns) in enumerate(frames)
    ]


def metadata_fps(frames: Sequence[CameraFrame]) -> float:
    """Estimate camera FPS from median metadata timestamp spacing."""

    if len(frames) < 2:
        raise ValueError("At least two camera frames are required to estimate FPS")
    deltas_ns = [
        frames[index].timestamp_ns - frames[index - 1].timestamp_ns
        for index in range(1, len(frames))
        if frames[index].timestamp_ns > frames[index - 1].timestamp_ns
    ]
    if not deltas_ns:
        raise ValueError("Camera metadata timestamps are not increasing")
    return 1_000_000_000.0 / statistics.median(deltas_ns)


def frame_window_for_stim(
    frames: Sequence[CameraFrame],
    *,
    stim_start_ms: float,
    stim_end_ms: float,
    pre_s: float,
    post_s: float,
) -> FrameWindow:
    """Map a stimulation window in pyControl ms to a camera frame range."""

    if not frames:
        raise ValueError("Cannot map stimulation window without camera frames")
    if pre_s < 0 or post_s < 0:
        raise ValueError("pre_s and post_s must be non-negative")
    if not (math.isfinite(stim_start_ms) and math.isfinite(stim_end_ms)):
        raise ValueError("Stimulation start/end must be finite")
    if stim_end_ms <= stim_start_ms:
        raise ValueError("Stimulation end must be after stimulation start")

    frame_times = [frame.relative_ms for frame in frames]
    fps = metadata_fps(frames)
    frame_period_ms = 1000.0 / fps
    camera_start_ms = frame_times[0]
    camera_end_ms = frame_times[-1] + frame_period_ms
    clip_start_ms = max(camera_start_ms, stim_start_ms - (pre_s * 1000.0))
    clip_end_ms = min(camera_end_ms, stim_end_ms + (post_s * 1000.0))
    if clip_end_ms <= clip_start_ms:
        raise ValueError("Requested clip is outside the camera timeline")

    start_index = bisect.bisect_left(frame_times, clip_start_ms)
    end_index = bisect.bisect_left(frame_times, clip_end_ms)
    if end_index <= start_index:
        end_index = min(len(frames), start_index + 1)
    if start_index >= len(frames):
        raise ValueError("Requested clip starts after the final camera frame")

    return FrameWindow(
        start_index=start_index,
        end_index_exclusive=end_index,
        clip_start_ms=clip_start_ms,
        clip_end_ms=clip_end_ms,
        stim_on_output_s=(stim_start_ms - clip_start_ms) / 1000.0,
        stim_off_output_s=(stim_end_ms - clip_start_ms) / 1000.0,
        fps=fps,
    )


def matched_summary_rows(summary_csv: str | Path) -> list[dict[str, str]]:
    """Load matched stimulation rows from a summary CSV."""

    with Path(summary_csv).open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if _bool_cell(row.get("matched_pycontrol_window"))
        ]


def session_summary_path(summary_root: str | Path, session_id: str) -> Path:
    """Return the expected per-session stimulation-motion summary path."""

    return Path(summary_root) / session_id / f"{session_id}_stim_motion_summary.csv"


def clip_output_path(
    session_dir: str | Path,
    row: dict[str, str],
    *,
    pre_s: float,
    post_s: float,
) -> Path:
    """Return the output path for one stimulation clip."""

    event_number = row.get("event_number", "")
    try:
        event_tag = f"{int(event_number):03d}"
    except (TypeError, ValueError):
        event_tag = sanitize_path_component(event_number, fallback="unknown")

    folder = amplitude_folder_name(row.get("amplitude_uA"), row.get("is_sham"))
    onset_index = sanitize_path_component(row.get("stim_event_index_onset", ""), fallback="unknown")
    filename = (
        f"event_{event_tag}_{folder}_onset_{onset_index}_"
        f"pre{seconds_tag(pre_s)}_post{seconds_tag(post_s)}.mp4"
    )
    return Path(session_dir) / "stim_videos" / folder / filename


def _escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:")


def _drawtext_font_option() -> str:
    for path in FONT_CANDIDATES:
        if path.exists():
            return f"fontfile={_escape_filter_path(path)}:"
    return ""


def _ffmpeg_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_annotation_filters(
    row: dict[str, str],
    window: FrameWindow,
    *,
    session_id: str,
) -> list[str]:
    """Build ffmpeg filters for trimming and video annotations."""

    if window.end_index_exclusive <= window.start_index:
        raise ValueError("Frame window has no frames")

    start = window.start_index
    end_inclusive = window.end_index_exclusive - 1
    fps = _ffmpeg_float(window.fps)
    stim_on = _ffmpeg_float(max(0.0, window.stim_on_output_s))
    stim_off = _ffmpeg_float(max(window.stim_on_output_s, window.stim_off_output_s))
    onset_flash_end = _ffmpeg_float(max(window.stim_on_output_s, window.stim_on_output_s + 0.35))
    offset_flash_end = _ffmpeg_float(max(window.stim_off_output_s, window.stim_off_output_s + 0.35))

    event_number = row.get("event_number", "")
    command = row.get("command") or amplitude_folder_name(row.get("amplitude_uA"), row.get("is_sham"))
    amplitude = row.get("amplitude_uA", "")
    sham_suffix = " SHAM" if _bool_cell(row.get("is_sham")) else ""
    line_1 = f"{session_id}  event {event_number}{sham_suffix}"
    line_2 = f"{command}  {amplitude} uA"
    active_color = "0x40A0FF@0.45" if _bool_cell(row.get("is_sham")) else "0xFF3030@0.45"
    font_option = _drawtext_font_option()

    rel_time = (
        "stim t=%{eif\\:(t-"
        f"{_ffmpeg_float(window.stim_on_output_s)})*1000\\:d}} ms"
    )

    return [
        f"select='between(n\\,{start}\\,{end_inclusive})'",
        f"setpts=N/({fps}*TB)",
        (
            "drawbox=x=0:y=0:w=iw:h=128:color=black@0.42:t=fill"
        ),
        (
            f"drawtext={font_option}text='{_escape_drawtext_text(line_1)}':"
            "x=22:y=18:fontsize=30:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=8"
        ),
        (
            f"drawtext={font_option}text='{_escape_drawtext_text(line_2)}':"
            "x=22:y=64:fontsize=26:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=8"
        ),
        (
            f"drawtext={font_option}text='{rel_time}':"
            "x=w-tw-22:y=22:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=8"
        ),
        "drawbox=x=0:y=ih-34:w=iw:h=34:color=black@0.55:t=fill",
        (
            "drawbox=x=0:y=ih-34:w=iw:h=34:"
            f"color={active_color}:t=fill:enable='between(t\\,{stim_on}\\,{stim_off})'"
        ),
        (
            f"drawtext={font_option}text='STIM ON':x=w-tw-22:y=68:fontsize=28:"
            "fontcolor=white:box=1:boxcolor=red@0.75:boxborderw=8:"
            f"enable='between(t\\,{stim_on}\\,{onset_flash_end})'"
        ),
        (
            f"drawtext={font_option}text='STIM OFF':x=w-tw-22:y=68:fontsize=28:"
            "fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=8:"
            f"enable='between(t\\,{stim_off}\\,{offset_flash_end})'"
        ),
    ]


def build_ffmpeg_command(
    input_video: str | Path,
    output_path: str | Path,
    row: dict[str, str],
    window: FrameWindow,
    *,
    session_id: str,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the ffmpeg command for one annotated clip."""

    filters = build_annotation_filters(row, window, session_id=session_id)
    fps = _ffmpeg_float(window.fps)
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        ",".join(filters),
        "-an",
        "-r",
        fps,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def prepare_session_clip_jobs(
    session_dir: str | Path,
    summary_csv: str | Path,
    *,
    camera_name: str,
    pre_s: float,
    post_s: float,
    ffmpeg: str = "ffmpeg",
) -> list[ClipJob]:
    """Prepare all matched clip extraction jobs for one session."""

    session_path = Path(session_dir)
    session_id = session_path.name
    camera_inputs = discover_camera_inputs(session_path, camera_name=camera_name)
    frames = load_camera_timeline(camera_inputs.metadata_path, camera_name=camera_name)

    jobs: list[ClipJob] = []
    for row in matched_summary_rows(summary_csv):
        start_ms = _as_float(row.get("window_start_ms"))
        end_ms = _as_float(row.get("window_end_ms"))
        frame_window = frame_window_for_stim(
            frames,
            stim_start_ms=start_ms,
            stim_end_ms=end_ms,
            pre_s=pre_s,
            post_s=post_s,
        )
        output_path = clip_output_path(session_path, row, pre_s=pre_s, post_s=post_s)
        ffmpeg_command = build_ffmpeg_command(
            camera_inputs.video_path,
            output_path,
            row,
            frame_window,
            session_id=session_id,
            ffmpeg=ffmpeg,
        )
        jobs.append(
            ClipJob(
                session_id=session_id,
                event_number=row.get("event_number", ""),
                amplitude_folder=output_path.parent.name,
                input_video=camera_inputs.video_path,
                output_path=output_path,
                frame_window=frame_window,
                ffmpeg_command=ffmpeg_command,
                summary_row=row,
            )
        )
    return jobs


def iter_session_dirs(animal_root: str | Path, sessions: Iterable[str] | None = None) -> list[Path]:
    """Return session directories from an animal root."""

    animal_path = Path(animal_root)
    if sessions:
        return [animal_path / session for session in sessions]
    return sorted(path for path in animal_path.iterdir() if path.is_dir())


def prepare_animal_clip_jobs(
    animal_root: str | Path,
    summary_root: str | Path,
    *,
    camera_name: str = "Camera_4",
    pre_s: float,
    post_s: float,
    sessions: Iterable[str] | None = None,
    ffmpeg: str = "ffmpeg",
) -> list[ClipJob]:
    """Prepare clip extraction jobs for an animal."""

    jobs: list[ClipJob] = []
    for session_dir in iter_session_dirs(animal_root, sessions):
        summary_csv = session_summary_path(summary_root, session_dir.name)
        if not summary_csv.exists():
            continue
        jobs.extend(
            prepare_session_clip_jobs(
                session_dir,
                summary_csv,
                camera_name=camera_name,
                pre_s=pre_s,
                post_s=post_s,
                ffmpeg=ffmpeg,
            )
        )
    return jobs


def run_clip_jobs(
    jobs: Sequence[ClipJob],
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> RunResult:
    """Run or print prepared clip extraction jobs."""

    written = 0
    skipped = 0
    for job in jobs:
        if job.output_path.exists() and not overwrite:
            skipped += 1
            print(f"skip existing: {job.output_path}")
            continue
        if dry_run:
            print(shlex.join(job.ffmpeg_command))
            continue
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(job.ffmpeg_command, check=True)
        written += 1
        print(f"wrote: {job.output_path}")
    return RunResult(prepared=len(jobs), written=written, skipped=skipped)
