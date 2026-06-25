"""Summarize motion sensor responses during stimulation windows."""

from __future__ import annotations

import ast
import csv
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PYCONTROL_STIM_ON_CODE = 101
PYCONTROL_STIM_OFF_CODE = 102
PYCONTROL_STIM_SHAM_CODE = 104
PYCONTROL_SESSION_START_CODE = 110
PYCONTROL_SESSION_END_CODE = 111
DEFAULT_MARKER_TOLERANCE_MS = 5000.0
PAA5100JE_HEIGHT_M = 0.02
PAA5100JE_CPI = 11.914 * (1 / PAA5100JE_HEIGHT_M)
CM_PER_MOTION_COUNT = 2.54 / PAA5100JE_CPI

SUMMARY_FIELDNAMES = [
    "session_id",
    "event_number",
    "command",
    "is_sham",
    "amplitude_uA",
    "duration_s",
    "shuffle_cycle",
    "stim_event_index_onset",
    "stim_event_index_offset",
    "window_start_ms",
    "window_end_ms",
    "matched_pycontrol_window",
    "motion_distance_cm",
    "motion_sample_count",
]


@dataclass(frozen=True)
class PyControlMarker:
    """One external stimulation marker printed by pyControl."""

    line_time_ms: float
    marker_time_ms: float
    code: int
    name: str
    count: int | None
    active: str


@dataclass(frozen=True)
class ExpectedMarker:
    """One expected pyControl marker derived from a stimulation attempt."""

    attempt_index: int
    role: str
    code: int
    expected_time_ms: float | None


@dataclass
class StimAttempt:
    """One stimulation attempt from the stimulation event log."""

    onset_row: dict[str, str]
    offset_row: dict[str, str] | None
    details: dict[str, object]
    event_number: int | None
    is_sham: bool
    expected_markers: list[ExpectedMarker]


@dataclass(frozen=True)
class PyControlSessionWindow:
    """The pyControl task time span to include in analysis."""

    start_ms: float
    end_ms: float


def _blank_for_nan(value: float | int | str | None) -> str | float | int:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _as_float(value: object) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_json_cell(value: str) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_datetime(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _parse_pycontrol_start_date(log_path: Path) -> dt.datetime | None:
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("I Start date"):
                continue
            _, _, value = line.partition(":")
            text = value.strip()
            try:
                return dt.datetime.strptime(text, "%Y/%m/%d %H:%M:%S")
            except ValueError:
                return None
    return None


def _relative_ms(wall_time_iso: str, start_date: dt.datetime | None) -> float | None:
    if start_date is None:
        return None
    parsed = _parse_datetime(wall_time_iso)
    if parsed is None:
        return None
    return (parsed - start_date).total_seconds() * 1000.0


def parse_numeric_sequence(value: str | int | float | None) -> list[float]:
    """Parse a scalar or Python/JSON-style list of numbers."""

    if value in (None, ""):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]

    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            return [float(text)]
        except ValueError:
            return []

    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    if isinstance(parsed, (list, tuple)):
        values = []
        for item in parsed:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        return values
    return []


def amplitude_mA_to_uA(value: str | int | float | None) -> float:
    """Return the scalar stimulation amplitude in microamps.

    Multi-channel stimulation logs store one amplitude per channel; the scalar
    condition amplitude is the maximum channel amplitude.
    """

    values = parse_numeric_sequence(value)
    if not values:
        return math.nan
    return max(values) * 1000.0


def load_analog_pca(file_path: str | Path) -> np.ndarray:
    """Load a pyControl analog ``.pca`` file as ``(time_ms, value)`` rows."""

    path = Path(file_path)
    data = np.fromfile(path, dtype="<i4")
    if data.size % 2:
        raise ValueError(f"Analog file has an odd number of int32 values: {path}")
    return data.reshape(-1, 2)


def motion_counts_to_cm(counts: float | int | np.ndarray) -> float | np.ndarray:
    """Convert PAA5100JE optical-flow counts to centimeters."""

    return counts * CM_PER_MOTION_COUNT


def summarize_motion_window(
    x_data: np.ndarray,
    y_data: np.ndarray,
    start_ms: float,
    end_ms: float,
) -> dict[str, float | int]:
    """Summarize paired MotSen1 X/Y samples inside ``[start_ms, end_ms)``."""

    if not (math.isfinite(start_ms) and math.isfinite(end_ms)) or end_ms <= start_ms:
        return {
            "motion_distance_cm": math.nan,
            "motion_sample_count": 0,
        }

    if x_data.size == 0 or y_data.size == 0:
        x_values = np.asarray([], dtype=float)
        y_values = np.asarray([], dtype=float)
    elif len(x_data) == len(y_data) and np.array_equal(x_data[:, 0], y_data[:, 0]):
        mask = (x_data[:, 0] >= start_ms) & (x_data[:, 0] < end_ms)
        x_values = x_data[mask, 1].astype(float)
        y_values = y_data[mask, 1].astype(float)
    else:
        y_by_time = {int(time_ms): float(value) for time_ms, value in y_data}
        x_window = x_data[(x_data[:, 0] >= start_ms) & (x_data[:, 0] < end_ms)]
        paired = [
            (float(x_value), y_by_time[int(time_ms)])
            for time_ms, x_value in x_window
            if int(time_ms) in y_by_time
        ]
        if paired:
            x_values, y_values = (np.asarray(values, dtype=float) for values in zip(*paired))
        else:
            x_values = np.asarray([], dtype=float)
            y_values = np.asarray([], dtype=float)

    magnitude = np.sqrt((x_values * x_values) + (y_values * y_values))
    return {
        "motion_distance_cm": float(motion_counts_to_cm(float(np.sum(magnitude)))),
        "motion_sample_count": int(x_values.size),
    }


_MARKER_RE = re.compile(
    r"^P\s+"
    r"(?P<line_time>-?\d+(?:\.\d+)?)\s+"
    r"(?:(?P<marker_time>-?\d+(?:\.\d+)?),\s*)?"
    r"external_stim_marker\s+"
    r"code=(?P<code>\d+)\s+"
    r"name=(?P<name>\S+)\s+"
    r"count=(?P<count>\d+)\s+"
    r"active=(?P<active>\S+)"
)

_PYCONTROL_TIMED_LINE_RE = re.compile(r"^[DP]\s+(?P<time>-?\d+(?:\.\d+)?)\b")


def parse_pycontrol_markers(log_path: str | Path) -> list[PyControlMarker]:
    """Parse external stimulation marker print lines from a pyControl log."""

    markers: list[PyControlMarker] = []
    path = Path(log_path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = _MARKER_RE.match(raw_line.strip())
            if not match:
                continue
            line_time = float(match.group("line_time"))
            marker_time = match.group("marker_time")
            markers.append(
                PyControlMarker(
                    line_time_ms=line_time,
                    marker_time_ms=float(marker_time) if marker_time is not None else line_time,
                    code=int(match.group("code")),
                    name=match.group("name"),
                    count=int(match.group("count")) if match.group("count") else None,
                    active=match.group("active"),
                )
            )
    return markers


def _parse_pycontrol_timestamps(log_path: str | Path) -> list[float]:
    """Return pyControl task timestamps from printed/event lines."""

    timestamps: list[float] = []
    path = Path(log_path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = _PYCONTROL_TIMED_LINE_RE.match(raw_line.strip())
            if match:
                timestamps.append(float(match.group("time")))
    return timestamps


def pycontrol_session_window(
    log_path: str | Path,
    pycontrol_markers: list[PyControlMarker],
    x_data: np.ndarray | None = None,
    y_data: np.ndarray | None = None,
) -> PyControlSessionWindow:
    """Return the pyControl task window in pyControl milliseconds.

    If explicit external session markers are present, use them. Otherwise fall
    back to the observed pyControl log and motion-data time range.
    """

    timestamps = _parse_pycontrol_timestamps(log_path)
    if x_data is not None and x_data.size:
        timestamps.extend(float(value) for value in x_data[:, 0])
    if y_data is not None and y_data.size:
        timestamps.extend(float(value) for value in y_data[:, 0])

    start_markers = [
        marker.marker_time_ms
        for marker in pycontrol_markers
        if marker.code == PYCONTROL_SESSION_START_CODE
    ]
    end_markers = [
        marker.marker_time_ms
        for marker in pycontrol_markers
        if marker.code == PYCONTROL_SESSION_END_CODE
    ]

    start_ms = start_markers[0] if start_markers else (min(timestamps) if timestamps else 0.0)
    end_ms = end_markers[-1] if end_markers else (max(timestamps) if timestamps else math.inf)
    return PyControlSessionWindow(start_ms=start_ms, end_ms=end_ms)


def _event_number_from_details(details: dict[str, object]) -> int | None:
    value = details.get("event_number")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_stim_attempts(
    stim_events_path: str | Path,
    *,
    pycontrol_start_date: dt.datetime | None = None,
) -> list[StimAttempt]:
    """Parse one stimulation attempt per ``stim_on`` or ``stim_sham`` row."""

    path = Path(stim_events_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    offset_by_event_number: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("event") != "stim_off":
            continue
        details = _parse_json_cell(row.get("details_json", ""))
        event_number = _event_number_from_details(details)
        if event_number is not None and event_number not in offset_by_event_number:
            offset_by_event_number[event_number] = row

    attempts: list[StimAttempt] = []
    for row in rows:
        event = row.get("event", "")
        if event not in {"stim_on", "stim_sham"}:
            continue
        details = _parse_json_cell(row.get("details_json", ""))
        event_number = _event_number_from_details(details)
        offset_row = offset_by_event_number.get(event_number) if event_number is not None else None
        is_sham = event == "stim_sham" or amplitude_mA_to_uA(row.get("amp_mA")) == 0.0
        attempt_index = len(attempts)

        onset_expected_ms = _relative_ms(row.get("wall_time_iso", ""), pycontrol_start_date)
        if is_sham:
            expected_markers = [
                ExpectedMarker(
                    attempt_index=attempt_index,
                    role="sham",
                    code=PYCONTROL_STIM_SHAM_CODE,
                    expected_time_ms=onset_expected_ms,
                )
            ]
        else:
            offset_expected_ms = (
                _relative_ms(offset_row.get("wall_time_iso", ""), pycontrol_start_date)
                if offset_row
                else None
            )
            expected_markers = [
                ExpectedMarker(
                    attempt_index=attempt_index,
                    role="on",
                    code=PYCONTROL_STIM_ON_CODE,
                    expected_time_ms=onset_expected_ms,
                ),
                ExpectedMarker(
                    attempt_index=attempt_index,
                    role="off",
                    code=PYCONTROL_STIM_OFF_CODE,
                    expected_time_ms=offset_expected_ms,
                ),
            ]

        attempts.append(
            StimAttempt(
                onset_row=row,
                offset_row=offset_row,
                details=details,
                event_number=event_number,
                is_sham=is_sham,
                expected_markers=expected_markers,
            )
        )

    return attempts


def _attempt_expected_onset_ms(attempt: StimAttempt) -> float | None:
    for marker in attempt.expected_markers:
        if marker.role in {"on", "sham"}:
            return marker.expected_time_ms
    return None


def _reindex_attempts(attempts: list[StimAttempt]) -> list[StimAttempt]:
    reindexed: list[StimAttempt] = []
    for attempt_index, attempt in enumerate(attempts):
        expected_markers = [
            ExpectedMarker(
                attempt_index=attempt_index,
                role=marker.role,
                code=marker.code,
                expected_time_ms=marker.expected_time_ms,
            )
            for marker in attempt.expected_markers
        ]
        reindexed.append(
            StimAttempt(
                onset_row=attempt.onset_row,
                offset_row=attempt.offset_row,
                details=attempt.details,
                event_number=attempt.event_number,
                is_sham=attempt.is_sham,
                expected_markers=expected_markers,
            )
        )
    return reindexed


def filter_attempts_to_pycontrol_session(
    attempts: list[StimAttempt],
    pycontrol_markers: list[PyControlMarker],
    session_window: PyControlSessionWindow,
) -> list[StimAttempt]:
    """Keep attempts whose expected onset falls inside the pyControl session."""

    time_offset_ms = _estimate_marker_time_offset_ms(
        [marker for attempt in attempts for marker in attempt.expected_markers],
        pycontrol_markers,
    )
    filtered: list[StimAttempt] = []
    for attempt in attempts:
        expected_onset_ms = _attempt_expected_onset_ms(attempt)
        if expected_onset_ms is None:
            filtered.append(attempt)
            continue
        pycontrol_onset_ms = expected_onset_ms + time_offset_ms
        if session_window.start_ms <= pycontrol_onset_ms <= session_window.end_ms:
            filtered.append(attempt)
    return _reindex_attempts(filtered)


def _estimate_marker_time_offset_ms(
    expected_markers: list[ExpectedMarker],
    pycontrol_markers: list[PyControlMarker],
    *,
    max_error_ms: float = 3000.0,
) -> float:
    offsets: list[float] = []
    for expected in expected_markers:
        if expected.expected_time_ms is None:
            continue
        same_code = [
            marker
            for marker in pycontrol_markers
            if marker.code == expected.code
            and abs(marker.marker_time_ms - expected.expected_time_ms) <= max_error_ms
        ]
        if not same_code:
            continue
        nearest = min(
            same_code,
            key=lambda marker: abs(marker.marker_time_ms - expected.expected_time_ms),
        )
        offsets.append(nearest.marker_time_ms - expected.expected_time_ms)
    if not offsets:
        return 0.0
    return float(np.median(np.asarray(offsets, dtype=float)))


def align_pycontrol_markers(
    attempts: list[StimAttempt],
    pycontrol_markers: list[PyControlMarker],
    *,
    tolerance_ms: float = DEFAULT_MARKER_TOLERANCE_MS,
) -> dict[tuple[int, str], PyControlMarker]:
    """Align stimulation-log expected markers to observed pyControl markers."""

    expected_markers = [marker for attempt in attempts for marker in attempt.expected_markers]
    time_offset_ms = _estimate_marker_time_offset_ms(expected_markers, pycontrol_markers)
    aligned: dict[tuple[int, str], PyControlMarker] = {}
    marker_index = 0

    for expected in expected_markers:
        if expected.expected_time_ms is None:
            continue
        target_ms = expected.expected_time_ms + time_offset_ms

        while (
            marker_index < len(pycontrol_markers)
            and pycontrol_markers[marker_index].marker_time_ms < target_ms - tolerance_ms
        ):
            marker_index += 1

        candidate_index = marker_index
        while candidate_index < len(pycontrol_markers):
            marker = pycontrol_markers[candidate_index]
            if marker.marker_time_ms > target_ms + tolerance_ms:
                break
            if marker.code == expected.code:
                aligned[(expected.attempt_index, expected.role)] = marker
                marker_index = candidate_index + 1
                break
            candidate_index += 1

    return aligned


def _motion_files_for_session(session_dir: Path, pycontrol_log_path: Path) -> tuple[Path, Path]:
    stem = pycontrol_log_path.with_suffix("").name
    x_candidates = sorted(session_dir.glob(f"{stem}_MotSen1-X.pca"))
    y_candidates = sorted(session_dir.glob(f"{stem}_MotSen1-Y.pca"))
    if not x_candidates:
        x_candidates = sorted(session_dir.glob("*_MotSen1-X.pca"))
    if not y_candidates:
        y_candidates = sorted(session_dir.glob("*_MotSen1-Y.pca"))
    if not x_candidates or not y_candidates:
        raise FileNotFoundError(f"Could not find MotSen1 X/Y .pca files in {session_dir}")
    return x_candidates[0], y_candidates[0]


def _pycontrol_log_for_session(session_dir: Path) -> Path:
    candidates = sorted(
        path for path in session_dir.glob("*.txt") if "missed_samples" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find pyControl .txt log in {session_dir}")
    return candidates[0]


def load_session_motion_data(session_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load paired MotSen1 X/Y analog data for a pyControl session folder."""

    session_path = Path(session_dir)
    pycontrol_log_path = _pycontrol_log_for_session(session_path)
    x_path, y_path = _motion_files_for_session(session_path, pycontrol_log_path)
    return load_analog_pca(x_path), load_analog_pca(y_path)


def _duration_seconds(row: dict[str, str]) -> float:
    duration = _as_float(row.get("duration_s"))
    return duration if math.isfinite(duration) else math.nan


def _match_window_for_attempt(
    attempt_index: int,
    attempt: StimAttempt,
    aligned_markers: dict[tuple[int, str], PyControlMarker],
) -> tuple[float, float, float, float, bool, str]:
    duration_s = _duration_seconds(attempt.onset_row)
    if attempt.is_sham:
        sham_marker = aligned_markers.get((attempt_index, "sham"))
        if sham_marker is None:
            return math.nan, math.nan, math.nan, math.nan, False, "missing_pycontrol_sham"
        window_start = sham_marker.marker_time_ms
        window_end = window_start + (duration_s * 1000.0)
        return (
            sham_marker.marker_time_ms,
            math.nan,
            window_start,
            window_end,
            True,
            "matched_sham_duration",
        )

    on_marker = aligned_markers.get((attempt_index, "on"))
    off_marker = aligned_markers.get((attempt_index, "off"))
    if on_marker is None and off_marker is None:
        return math.nan, math.nan, math.nan, math.nan, False, "missing_pycontrol_on_off"
    if on_marker is None:
        return math.nan, off_marker.marker_time_ms, math.nan, math.nan, False, "missing_pycontrol_on"
    if off_marker is None:
        return on_marker.marker_time_ms, math.nan, math.nan, math.nan, False, "missing_pycontrol_off"
    return (
        on_marker.marker_time_ms,
        off_marker.marker_time_ms,
        on_marker.marker_time_ms,
        off_marker.marker_time_ms,
        True,
        "matched",
    )


def summarize_session(
    session_dir: str | Path,
    *,
    animal_id: str | None = None,
) -> list[dict[str, object]]:
    """Return one summary row per stimulation attempt in ``session_dir``."""

    session_path = Path(session_dir)
    session_id = session_path.name
    stim_events_path = session_path / "stim_events.csv"
    if not stim_events_path.exists():
        raise FileNotFoundError(f"Could not find stim_events.csv in {session_path}")

    pycontrol_log_path = _pycontrol_log_for_session(session_path)
    pycontrol_start_date = _parse_pycontrol_start_date(pycontrol_log_path)
    attempts = parse_stim_attempts(
        stim_events_path,
        pycontrol_start_date=pycontrol_start_date,
    )
    pycontrol_markers = parse_pycontrol_markers(pycontrol_log_path)
    x_data, y_data = load_session_motion_data(session_path)
    session_window = pycontrol_session_window(
        pycontrol_log_path,
        pycontrol_markers,
        x_data,
        y_data,
    )
    attempts = filter_attempts_to_pycontrol_session(
        attempts,
        pycontrol_markers,
        session_window,
    )
    aligned_markers = align_pycontrol_markers(attempts, pycontrol_markers)

    rows: list[dict[str, object]] = []
    for attempt_index, attempt in enumerate(attempts):
        onset = attempt.onset_row
        offset = attempt.offset_row
        (
            _pycontrol_onset_ms,
            _pycontrol_offset_ms,
            window_start_ms,
            window_end_ms,
            matched_window,
            _match_status,
        ) = _match_window_for_attempt(attempt_index, attempt, aligned_markers)
        if matched_window:
            motion = summarize_motion_window(x_data, y_data, window_start_ms, window_end_ms)
        else:
            motion = {
                "motion_distance_cm": math.nan,
                "motion_sample_count": 0,
            }

        details = attempt.details
        row = {
            "session_id": session_id,
            "event_number": attempt.event_number if attempt.event_number is not None else "",
            "command": onset.get("command", ""),
            "is_sham": str(bool(attempt.is_sham)),
            "amplitude_uA": amplitude_mA_to_uA(onset.get("amp_mA")),
            "duration_s": onset.get("duration_s", ""),
            "shuffle_cycle": details.get("shuffle_cycle", ""),
            "stim_event_index_onset": onset.get("event_index", ""),
            "stim_event_index_offset": offset.get("event_index", "") if offset else "",
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "matched_pycontrol_window": str(bool(matched_window)),
            **motion,
        }
        rows.append({key: _blank_for_nan(row.get(key, "")) for key in SUMMARY_FIELDNAMES})

    return rows


def write_summary_csv(rows: Iterable[dict[str, object]], output_path: str | Path) -> Path:
    """Write summary rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDNAMES})
    return path


def session_summary_csv_name(session_id: str) -> str:
    """Return the session-specific stimulation-motion summary filename."""

    return f"{session_id}_stim_motion_summary.csv"


def summarize_animal(
    animal_root: str | Path,
    *,
    output_root: str | Path | None = None,
) -> tuple[list[dict[str, object]], list[Path]]:
    """Summarize all session folders for one animal and write CSV outputs."""

    animal_path = Path(animal_root)
    animal_id = animal_path.name
    resolved_output_root = (
        Path(output_root) if output_root is not None else Path("analysis") / "stim_motion" / animal_id
    )
    all_rows: list[dict[str, object]] = []
    written_paths: list[Path] = []

    session_dirs = sorted(
        path
        for path in animal_path.iterdir()
        if path.is_dir() and (path / "stim_events.csv").exists()
    )
    if not session_dirs:
        raise FileNotFoundError(f"No session folders with stim_events.csv found in {animal_path}")

    for session_dir in session_dirs:
        session_rows = summarize_session(session_dir, animal_id=animal_id)
        all_rows.extend(session_rows)
        session_output = (
            resolved_output_root
            / session_dir.name
            / session_summary_csv_name(session_dir.name)
        )
        written_paths.append(write_summary_csv(session_rows, session_output))

    combined_output = resolved_output_root / f"{animal_id}_stim_motion_summary.csv"
    written_paths.append(write_summary_csv(all_rows, combined_output))
    return all_rows, written_paths
