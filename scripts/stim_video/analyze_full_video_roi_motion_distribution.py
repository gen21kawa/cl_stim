"""Analyze ROI motion-energy distributions from full-session camera videos."""

from __future__ import annotations

import argparse
import bisect
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANIMAL_ROOT = PROJECT_ROOT / "data" / "raw" / "M114"

PERIODS = (
    ("pre_3", -9.0, -6.0),
    ("pre_2", -6.0, -3.0),
    ("pre_1", -3.0, 0.0),
    ("stim", 0.0, 3.0),
    ("post", 3.0, 6.0),
)
PERIOD_ORDER = [period[0] for period in PERIODS]
PLOT_PERIOD_ORDER = ["stim", "post", "pre_1", "pre_2", "pre_3"]
PERIOD_STYLE = {
    "pre_3": {"color": "#d9d9d9", "alpha": 0.45, "linewidth": 1.0},
    "pre_2": {"color": "#bdbdbd", "alpha": 0.55, "linewidth": 1.15},
    "pre_1": {"color": "#969696", "alpha": 0.7, "linewidth": 1.3},
    "post": {"color": "#fdae6b", "alpha": 0.85, "linewidth": 1.6},
    "stim": {"color": "#de2d26", "alpha": 1.0, "linewidth": 2.1},
}
ROI_BOXES = {
    "face": (200, 360, 200, 150),
    "forelimb": (200, 550, 280, 130),
}


@dataclass(frozen=True)
class CameraData:
    video_path: Path
    metadata_path: Path
    frame_times_ms: np.ndarray
    metadata_fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute full-video ROI motion-energy distributions around stimulation."
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
        help="Stim-motion summary root. Defaults to analysis/stim_motion/<animal_id>.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to analysis/video_motion_distribution/<animal_id>.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Session ID to process. May be passed more than once.",
    )
    parser.add_argument(
        "--camera-name",
        default="Camera_4",
        help="Camera name to analyze. Defaults to Camera_4.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional per-session trial limit for smoke tests.",
    )
    return parser.parse_args()


def _bool_cell(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def discover_camera_files(session_dir: Path, camera_name: str) -> tuple[Path, Path]:
    camera_dirs = sorted(path for path in session_dir.glob("*_cameras") if path.is_dir())
    video_candidates: list[Path] = []
    for camera_dir in camera_dirs:
        video_candidates.extend(
            sorted(
                path
                for path in camera_dir.iterdir()
                if path.is_file()
                and path.stem == camera_name
                and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
            )
        )
    if not video_candidates:
        video_candidates = sorted(
            path
            for path in session_dir.rglob(f"{camera_name}.*")
            if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
        )
    if not video_candidates:
        raise FileNotFoundError(f"Could not find {camera_name} video in {session_dir}")

    metadata_candidates = [
        *(camera_dir / "metadata.csv" for camera_dir in camera_dirs),
        session_dir / "metadata.csv",
        *sorted(session_dir.rglob("metadata.csv")),
    ]
    seen = set()
    for metadata_path in metadata_candidates:
        if metadata_path in seen or not metadata_path.exists():
            continue
        seen.add(metadata_path)
        if metadata_contains_camera(metadata_path, camera_name):
            return video_candidates[0], metadata_path
    raise FileNotFoundError(f"Could not find metadata.csv with {camera_name} rows")


def metadata_contains_camera(metadata_path: Path, camera_name: str) -> bool:
    reader = pd.read_csv(metadata_path, delimiter=";", usecols=["frame_camera_name"], chunksize=10000)
    return any((chunk["frame_camera_name"] == camera_name).any() for chunk in reader)


def load_camera_data(session_dir: Path, camera_name: str) -> CameraData:
    video_path, metadata_path = discover_camera_files(session_dir, camera_name)
    metadata = pd.read_csv(
        metadata_path,
        delimiter=";",
        usecols=["frame_camera_name", "frame_timestamp"],
    )
    metadata = metadata[metadata["frame_camera_name"] == camera_name].copy()
    if metadata.empty:
        raise ValueError(f"No {camera_name} rows in {metadata_path}")

    timestamps_ns = metadata["frame_timestamp"].to_numpy(dtype=np.int64)
    frame_times_ms = (timestamps_ns - timestamps_ns[0]) / 1_000_000.0
    deltas_ms = np.diff(frame_times_ms)
    deltas_ms = deltas_ms[deltas_ms > 0]
    if deltas_ms.size == 0:
        raise ValueError(f"Non-increasing camera timestamps in {metadata_path}")
    metadata_fps = 1000.0 / float(np.median(deltas_ms))
    return CameraData(
        video_path=video_path,
        metadata_path=metadata_path,
        frame_times_ms=frame_times_ms,
        metadata_fps=metadata_fps,
    )


def load_summary(summary_root: Path, session_id: str) -> pd.DataFrame:
    summary_path = summary_root / session_id / f"{session_id}_stim_motion_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing stimulation summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    summary = summary[summary["matched_pycontrol_window"].map(_bool_cell)].copy()
    summary["window_start_ms"] = summary["window_start_ms"].astype(float)
    summary["duration_s"] = summary["duration_s"].astype(float)
    summary["amplitude_uA"] = summary["amplitude_uA"].astype(float)
    return summary.sort_values("window_start_ms").reset_index(drop=True)


def period_frame_ranges(
    frame_times_ms: np.ndarray, stim_start_ms: float, metadata_fps: float
) -> tuple[dict[str, tuple[int, int]], str | None]:
    ranges: dict[str, tuple[int, int]] = {}
    expected_frames_per_period = int(round(3.0 * metadata_fps))
    minimum_frames_per_period = int(round(expected_frames_per_period * 0.98))
    for period, relative_start_s, relative_end_s in PERIODS:
        period_start_ms = stim_start_ms + (relative_start_s * 1000.0)
        period_end_ms = stim_start_ms + (relative_end_s * 1000.0)
        start_index = bisect.bisect_left(frame_times_ms, period_start_ms)
        end_index = bisect.bisect_left(frame_times_ms, period_end_ms)
        if start_index <= 0:
            return {}, f"{period} starts before enough previous camera frame is available"
        if end_index <= start_index:
            return {}, f"{period} has no frames"
        if end_index > len(frame_times_ms):
            return {}, f"{period} extends past the camera recording"
        if (end_index - start_index) < minimum_frames_per_period:
            return (
                {},
                f"{period} has only {end_index - start_index} frames; "
                f"expected about {expected_frames_per_period}",
            )
        ranges[period] = (start_index, end_index)
    return ranges, None


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)


def read_frame_sequence(
    cap: cv2.VideoCapture, start_index: int, end_index_inclusive: int
) -> dict[int, np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    frames: dict[int, np.ndarray] = {}
    for frame_index in range(start_index, end_index_inclusive + 1):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_index}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[frame_index] = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    return frames


def validate_rois(frame_width: int, frame_height: int) -> None:
    for roi_name, (x, y, width, height) in ROI_BOXES.items():
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid ROI {roi_name}: {ROI_BOXES[roi_name]}")
        if x + width > frame_width or y + height > frame_height:
            raise ValueError(
                f"ROI {roi_name} extends outside frame {frame_width} x {frame_height}: "
                f"{ROI_BOXES[roi_name]}"
            )


def crop_roi(gray_frame: np.ndarray, roi_name: str) -> np.ndarray:
    x, y, width, height = ROI_BOXES[roi_name]
    return gray_frame[y : y + height, x : x + width]


def compute_trial_motion_energy(
    cap: cv2.VideoCapture,
    frame_times_ms: np.ndarray,
    trial_row: pd.Series,
    ranges: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    stim_start_ms = float(trial_row["window_start_ms"])
    min_frame = min(start for start, _ in ranges.values()) - 1
    max_frame = max(end for _, end in ranges.values()) - 1
    frames = read_frame_sequence(cap, min_frame, max_frame)

    rows = []
    for period, relative_start_s, relative_end_s in PERIODS:
        start_index, end_index = ranges[period]
        for frame_index in range(start_index, end_index):
            previous = frames[frame_index - 1]
            current = frames[frame_index]
            frame_time_relative_to_stim_s = (frame_times_ms[frame_index] - stim_start_ms) / 1000.0
            for roi_name in ROI_BOXES:
                previous_roi = crop_roi(previous, roi_name)
                current_roi = crop_roi(current, roi_name)
                motion_energy = float(np.abs(current_roi - previous_roi).mean())
                rows.append(
                    {
                        "session_id": trial_row["session_id"],
                        "event_number": int(trial_row["event_number"]),
                        "amplitude_uA": float(trial_row["amplitude_uA"]),
                        "is_sham": bool(_bool_cell(trial_row["is_sham"])),
                        "roi": roi_name,
                        "period": period,
                        "period_start_relative_s": relative_start_s,
                        "period_end_relative_s": relative_end_s,
                        "frame_index": frame_index,
                        "frame_time_relative_to_stim_s": frame_time_relative_to_stim_s,
                        "motion_energy_per_pixel": motion_energy,
                    }
                )
    return pd.DataFrame(rows)


def period_totals(per_frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "session_id",
        "event_number",
        "amplitude_uA",
        "is_sham",
        "roi",
        "period",
        "period_start_relative_s",
        "period_end_relative_s",
    ]
    totals = (
        per_frame.groupby(group_cols, as_index=False)
        .agg(
            period_total_motion_energy_per_pixel=("motion_energy_per_pixel", "sum"),
            period_mean_motion_energy_per_pixel=("motion_energy_per_pixel", "mean"),
            n_frames=("frame_index", "nunique"),
        )
        .sort_values(["event_number", "roi", "period_start_relative_s"])
    )
    return totals


def period_summary(totals: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["session_id", "amplitude_uA", "is_sham", "roi", "period"]
    summary = (
        totals.groupby(group_cols, as_index=False)
        .agg(
            n_trials=("event_number", "nunique"),
            mean_period_total_motion_energy_per_pixel=(
                "period_total_motion_energy_per_pixel",
                "mean",
            ),
            median_period_total_motion_energy_per_pixel=(
                "period_total_motion_energy_per_pixel",
                "median",
            ),
            std_period_total_motion_energy_per_pixel=(
                "period_total_motion_energy_per_pixel",
                "std",
            ),
            p25_period_total_motion_energy_per_pixel=(
                "period_total_motion_energy_per_pixel",
                lambda values: float(np.nanpercentile(values, 25)),
            ),
            p75_period_total_motion_energy_per_pixel=(
                "period_total_motion_energy_per_pixel",
                lambda values: float(np.nanpercentile(values, 75)),
            ),
        )
        .sort_values(["session_id", "roi", "amplitude_uA"])
    )
    summary["period"] = pd.Categorical(summary["period"], PERIOD_ORDER, ordered=True)
    summary = summary.sort_values(["session_id", "roi", "amplitude_uA", "period"])

    ratio_rows = []
    for keys, roi_totals in totals.groupby(["session_id", "amplitude_uA", "is_sham", "roi"]):
        period_medians = roi_totals.groupby("period")[
            "period_total_motion_energy_per_pixel"
        ].median()
        stim = period_medians.get("stim", np.nan)
        pre_1 = period_medians.get("pre_1", np.nan)
        pre_values = roi_totals[roi_totals["period"].isin(["pre_3", "pre_2", "pre_1"])][
            "period_total_motion_energy_per_pixel"
        ]
        combined_pre = float(pre_values.median()) if not pre_values.empty else np.nan
        ratio_rows.append(
            {
                "session_id": keys[0],
                "amplitude_uA": keys[1],
                "is_sham": keys[2],
                "roi": keys[3],
                "stim_to_pre_1_median_ratio": stim / pre_1 if pre_1 else np.nan,
                "stim_to_combined_pre_median_ratio": (
                    stim / combined_pre if combined_pre else np.nan
                ),
            }
        )
    ratios = pd.DataFrame(ratio_rows)
    return summary.merge(
        ratios,
        on=["session_id", "amplitude_uA", "is_sham", "roi"],
        how="left",
    )


def period_palette() -> dict[str, str]:
    return {period: style["color"] for period, style in PERIOD_STYLE.items()}


def amplitude_label(value: float) -> str:
    if abs(value) < 1e-9:
        return "sham"
    return f"{int(value)} uA" if abs(value - round(value)) < 1e-9 else f"{value:g} uA"


def plot_kde_line(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    style: dict[str, float | str],
    label: str,
    x_limit: float | None,
) -> None:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.nanstd(values) == 0:
        return
    upper = x_limit if x_limit is not None else float(np.nanpercentile(values, 99.8))
    if not np.isfinite(upper) or upper <= 0:
        upper = float(np.nanmax(values))
    if not np.isfinite(upper) or upper <= 0:
        return

    grid = np.linspace(0, upper, 256)
    try:
        density = stats.gaussian_kde(values)(grid)
    except (np.linalg.LinAlgError, ValueError):
        return
    ax.plot(
        grid,
        density,
        color=str(style["color"]),
        alpha=float(style["alpha"]),
        linewidth=float(style["linewidth"]),
        label=label,
    )


def jittered_x(center: float, n_points: int, seed: int) -> np.ndarray:
    if n_points == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-0.18, 0.18, size=n_points)


def plot_per_frame_distribution(
    per_frame: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    amplitudes = sorted(per_frame["amplitude_uA"].dropna().unique())
    rois = [roi for roi in ROI_BOXES if roi in set(per_frame["roi"])]
    fig, axes = plt.subplots(
        len(rois),
        len(amplitudes),
        figsize=(3.4 * len(amplitudes), 3.0 * len(rois)),
        squeeze=False,
        sharey=False,
    )
    for row_index, roi in enumerate(rois):
        for col_index, amplitude in enumerate(amplitudes):
            ax = axes[row_index, col_index]
            subset = per_frame[
                (per_frame["roi"] == roi) & (per_frame["amplitude_uA"] == amplitude)
            ]
            if subset.empty:
                ax.axis("off")
                continue
            x_limit = float(np.nanpercentile(subset["motion_energy_per_pixel"], 99.5))
            if not np.isfinite(x_limit) or x_limit <= 0:
                x_limit = None
            for period in PLOT_PERIOD_ORDER:
                values = subset.loc[
                    subset["period"] == period, "motion_energy_per_pixel"
                ].to_numpy()
                style = PERIOD_STYLE[period]
                plot_kde_line(
                    ax,
                    values,
                    style=style,
                    label=period,
                    x_limit=x_limit,
                )
            if x_limit is not None:
                ax.set_xlim(0, x_limit)
            if row_index == 0:
                ax.set_title(amplitude_label(amplitude))
            if col_index == 0:
                ax.set_ylabel(f"{roi}\nDensity")
            else:
                ax.set_ylabel("")
            if row_index == len(rois) - 1:
                ax.set_xlabel("Motion energy per pixel")
            else:
                ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, title="Period", loc="upper right")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def add_period_total_means(ax: plt.Axes, subset: pd.DataFrame) -> None:
    for x_pos, period in enumerate(PLOT_PERIOD_ORDER):
        values = subset.loc[
            subset["period"] == period, "period_total_motion_energy_per_pixel"
        ]
        if values.empty:
            continue
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        ax.errorbar(
            x_pos,
            mean,
            yerr=std,
            fmt="_",
            color="black",
            markersize=16,
            capsize=4,
            linewidth=1.2,
        )


def plot_period_total_distribution(
    totals: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    amplitudes = sorted(totals["amplitude_uA"].dropna().unique())
    rois = [roi for roi in ROI_BOXES if roi in set(totals["roi"])]
    plot_df = totals.copy()
    plot_df["period"] = pd.Categorical(plot_df["period"], PLOT_PERIOD_ORDER, ordered=True)
    fig, axes = plt.subplots(
        len(rois),
        len(amplitudes),
        figsize=(3.4 * len(amplitudes), 3.0 * len(rois)),
        squeeze=False,
        sharey=False,
    )
    for row_index, roi in enumerate(rois):
        for col_index, amplitude in enumerate(amplitudes):
            ax = axes[row_index, col_index]
            subset = plot_df[
                (plot_df["roi"] == roi) & (plot_df["amplitude_uA"] == amplitude)
            ]
            if subset.empty:
                ax.axis("off")
                continue
            for x_pos, period in enumerate(PLOT_PERIOD_ORDER):
                values = subset.loc[
                    subset["period"] == period,
                    "period_total_motion_energy_per_pixel",
                ].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                seed = int((row_index + 1) * 10000 + (col_index + 1) * 100 + x_pos)
                ax.scatter(
                    jittered_x(float(x_pos), len(values), seed),
                    values,
                    color=PERIOD_STYLE[period]["color"],
                    alpha=0.85,
                    s=12,
                    linewidth=0,
                )
            add_period_total_means(ax, subset)
            ax.set_xticks(range(len(PLOT_PERIOD_ORDER)))
            ax.set_xticklabels(PLOT_PERIOD_ORDER)
            if row_index == 0:
                ax.set_title(amplitude_label(amplitude))
            if col_index == 0:
                ax.set_ylabel(f"{roi}\nTotal ME per pixel")
            else:
                ax.set_ylabel("")
            if row_index == len(rois) - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=35)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_session_outputs(
    session_output: Path,
    session_id: str,
    per_frame: pd.DataFrame,
    totals: pd.DataFrame,
    summary: pd.DataFrame,
    dropped: pd.DataFrame,
) -> None:
    session_output.mkdir(parents=True, exist_ok=True)
    per_frame.to_csv(
        session_output / f"{session_id}_roi_motion_energy_full_video_per_frame.csv",
        index=False,
    )
    totals.to_csv(
        session_output / f"{session_id}_roi_motion_energy_full_video_period_totals.csv",
        index=False,
    )
    summary.to_csv(
        session_output / f"{session_id}_roi_motion_energy_period_summary.csv",
        index=False,
    )
    dropped.to_csv(session_output / f"{session_id}_dropped_trials.csv", index=False)
    plot_per_frame_distribution(
        per_frame,
        session_output / f"{session_id}_per_frame_motion_energy_distribution",
        title=f"{session_id}: per-frame ROI motion energy",
    )
    plot_period_total_distribution(
        totals,
        session_output / f"{session_id}_period_total_motion_energy_distribution",
        title=f"{session_id}: period-total ROI motion energy",
    )


def analyze_session(
    animal_root: Path,
    summary_root: Path,
    output_root: Path,
    session_id: str,
    camera_name: str,
    *,
    max_trials: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session_dir = animal_root / session_id
    camera = load_camera_data(session_dir, camera_name)
    summary = load_summary(summary_root, session_id)
    if max_trials is not None:
        summary = summary.head(max_trials).copy()

    cap = cv2.VideoCapture(str(camera.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {camera.video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    validate_rois(frame_width, frame_height)
    if abs(frame_count - len(camera.frame_times_ms)) > 1:
        print(
            f"Warning: metadata rows ({len(camera.frame_times_ms)}) and video frames "
            f"({frame_count}) differ for {session_id}",
            file=sys.stderr,
        )

    per_trial_frames: list[pd.DataFrame] = []
    dropped_rows = []
    for trial_index, trial_row in summary.iterrows():
        ranges, reason = period_frame_ranges(
            camera.frame_times_ms,
            float(trial_row["window_start_ms"]),
            camera.metadata_fps,
        )
        if reason is not None:
            dropped_rows.append(
                {
                    "session_id": session_id,
                    "event_number": int(trial_row["event_number"]),
                    "amplitude_uA": float(trial_row["amplitude_uA"]),
                    "is_sham": bool(_bool_cell(trial_row["is_sham"])),
                    "drop_reason": reason,
                }
            )
            continue
        per_trial_frames.append(
            compute_trial_motion_energy(cap, camera.frame_times_ms, trial_row, ranges)
        )
        if (len(per_trial_frames) % 10) == 0:
            print(
                f"  {session_id}: {len(per_trial_frames)} kept trials processed",
                flush=True,
            )

    cap.release()
    if per_trial_frames:
        per_frame = pd.concat(per_trial_frames, ignore_index=True)
    else:
        per_frame = pd.DataFrame()
    totals = period_totals(per_frame) if not per_frame.empty else pd.DataFrame()
    summary_table = period_summary(totals) if not totals.empty else pd.DataFrame()
    dropped = pd.DataFrame(dropped_rows)

    session_output = output_root / session_id
    write_session_outputs(session_output, session_id, per_frame, totals, summary_table, dropped)
    metadata_summary = pd.DataFrame(
        [
            {
                "session_id": session_id,
                "camera_name": camera_name,
                "video_path": str(camera.video_path),
                "metadata_path": str(camera.metadata_path),
                "metadata_fps": camera.metadata_fps,
                "metadata_frame_count": len(camera.frame_times_ms),
                "video_frame_count": frame_count,
                "kept_trials": int(per_frame["event_number"].nunique())
                if not per_frame.empty
                else 0,
                "dropped_trials": len(dropped),
            }
        ]
    )
    metadata_summary.to_csv(
        session_output / f"{session_id}_full_video_motion_energy_metadata.csv",
        index=False,
    )
    print(
        f"{session_id}: metadata FPS {camera.metadata_fps:.3f}; "
        f"kept {metadata_summary['kept_trials'].iloc[0]} trials; "
        f"dropped {len(dropped)}",
        flush=True,
    )
    return per_frame, totals, summary_table, dropped


def session_ids(animal_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(path.name for path in animal_root.iterdir() if path.is_dir())


def write_combined_outputs(
    output_root: Path,
    animal_id: str,
    per_frame_tables: list[pd.DataFrame],
    total_tables: list[pd.DataFrame],
    summary_tables: list[pd.DataFrame],
    dropped_tables: list[pd.DataFrame],
) -> None:
    combined_output = output_root / "all_sessions"
    combined_output.mkdir(parents=True, exist_ok=True)
    if per_frame_tables:
        per_frame = pd.concat(per_frame_tables, ignore_index=True)
        per_frame.to_csv(
            combined_output / f"{animal_id}_roi_motion_energy_full_video_per_frame.csv",
            index=False,
        )
        plot_per_frame_distribution(
            per_frame,
            combined_output / f"{animal_id}_per_frame_motion_energy_distribution",
            title=f"{animal_id}: per-frame ROI motion energy",
        )
    if total_tables:
        totals = pd.concat(total_tables, ignore_index=True)
        totals.to_csv(
            combined_output / f"{animal_id}_roi_motion_energy_full_video_period_totals.csv",
            index=False,
        )
        plot_period_total_distribution(
            totals,
            combined_output / f"{animal_id}_period_total_motion_energy_distribution",
            title=f"{animal_id}: period-total ROI motion energy",
        )
    if summary_tables:
        pd.concat(summary_tables, ignore_index=True).to_csv(
            combined_output / f"{animal_id}_roi_motion_energy_period_summary.csv",
            index=False,
        )
    if dropped_tables:
        pd.concat(dropped_tables, ignore_index=True).to_csv(
            combined_output / f"{animal_id}_dropped_trials.csv",
            index=False,
        )


def main() -> int:
    args = parse_args()
    animal_root = args.animal_root
    animal_id = animal_root.name
    summary_root = args.summary_root or PROJECT_ROOT / "analysis" / "stim_motion" / animal_id
    output_root = (
        args.output_root
        if args.output_root is not None
        else PROJECT_ROOT / "analysis" / "video_motion_distribution" / animal_id
    )

    output_root.mkdir(parents=True, exist_ok=True)
    per_frame_tables = []
    total_tables = []
    summary_tables = []
    dropped_tables = []
    for session_id in session_ids(animal_root, args.session):
        per_frame, totals, summary_table, dropped = analyze_session(
            animal_root,
            summary_root,
            output_root,
            session_id,
            args.camera_name,
            max_trials=args.max_trials,
        )
        if not per_frame.empty:
            per_frame_tables.append(per_frame)
        if not totals.empty:
            total_tables.append(totals)
        if not summary_table.empty:
            summary_tables.append(summary_table)
        if not dropped.empty:
            dropped_tables.append(dropped)
    write_combined_outputs(
        output_root,
        animal_id,
        per_frame_tables,
        total_tables,
        summary_tables,
        dropped_tables,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
