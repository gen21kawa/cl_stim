"""Estimate ROI motion-response latencies from motion-energy CSVs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANIMAL_ID = "M114"


@dataclass(frozen=True)
class LatencyParams:
    stim_onset_s: float
    response_end_s: float
    baseline_start_s: float
    baseline_end_s: float
    threshold_k: float
    min_duration_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate per-trial ROI movement latencies from motion-energy CSVs."
    )
    parser.add_argument(
        "--animal-id",
        default=DEFAULT_ANIMAL_ID,
        help=f"Animal ID. Defaults to {DEFAULT_ANIMAL_ID}.",
    )
    parser.add_argument(
        "--motion-root",
        type=Path,
        default=None,
        help="Folder containing per-session video-motion-energy outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Folder for latency CSVs and plots.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Session ID to process. May be passed more than once.",
    )
    parser.add_argument("--stim-onset-s", type=float, default=2.0)
    parser.add_argument("--response-end-s", type=float, default=5.0)
    parser.add_argument("--baseline-start-s", type=float, default=0.5)
    parser.add_argument("--baseline-end-s", type=float, default=1.9)
    parser.add_argument("--threshold-k", type=float, default=4.0)
    parser.add_argument("--min-duration-ms", type=float, default=50.0)
    return parser.parse_args()


def robust_threshold(values: np.ndarray, k: float) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma == 0:
        sigma = float(np.nanstd(values))
    threshold = median + (k * sigma)
    return median, sigma, threshold


def estimate_sample_period_s(time_s: np.ndarray) -> float:
    diffs = np.diff(np.sort(time_s[np.isfinite(time_s)]))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return np.nan
    return float(np.nanmedian(diffs))


def first_sustained_crossing(
    time_s: np.ndarray, signal: np.ndarray, threshold: float, min_frames: int
) -> float:
    if not np.isfinite(threshold) or min_frames <= 0:
        return np.nan

    above = np.asarray(signal) > threshold
    for idx in range(0, len(above) - min_frames + 1):
        if bool(np.all(above[idx : idx + min_frames])):
            return float(time_s[idx])
    return np.nan


def roi_names_from_columns(columns: list[str]) -> list[str]:
    rois = [
        col.removesuffix("_motion_energy_per_pixel")
        for col in columns
        if col.endswith("_motion_energy_per_pixel")
    ]
    preferred = [roi for roi in ["face", "forelimb"] if roi in rois]
    return preferred + [roi for roi in rois if roi not in preferred]


def amplitude_label(stim_amp_uA: int | float) -> str:
    amp = int(stim_amp_uA)
    return "sham" if amp == 0 else f"{amp} uA"


def compute_latency_table(
    energy_df: pd.DataFrame, rois: list[str], params: LatencyParams
) -> pd.DataFrame:
    rows = []
    group_cols = ["session_name", "stim_amp_uA", "video_dir_name", "video"]

    for group_values, trial_df in energy_df.groupby(group_cols, dropna=False):
        group_info = dict(zip(group_cols, group_values))
        trial_df = trial_df.sort_values("time_s")
        time_s = trial_df["time_s"].to_numpy(dtype=float)
        sample_period_s = estimate_sample_period_s(time_s)
        if np.isfinite(sample_period_s) and sample_period_s > 0:
            min_frames = max(1, int(round((params.min_duration_ms / 1000.0) / sample_period_s)))
        else:
            min_frames = 1

        baseline_mask = (
            (trial_df["time_s"] >= params.baseline_start_s)
            & (trial_df["time_s"] < params.baseline_end_s)
        )
        response_mask = (
            (trial_df["time_s"] >= params.stim_onset_s)
            & (trial_df["time_s"] <= params.response_end_s)
        )

        for roi in rois:
            col = f"{roi}_motion_energy_per_pixel"
            baseline_values = trial_df.loc[baseline_mask, col].to_numpy(dtype=float)
            baseline_median, baseline_sigma, threshold = robust_threshold(
                baseline_values, params.threshold_k
            )

            response_time = trial_df.loc[response_mask, "time_s"].to_numpy(dtype=float)
            response_signal = trial_df.loc[response_mask, col].to_numpy(dtype=float)
            onset_time_s = first_sustained_crossing(
                response_time, response_signal, threshold, min_frames
            )
            latency_s = onset_time_s - params.stim_onset_s if np.isfinite(onset_time_s) else np.nan

            baseline_above_fraction = (
                float(np.mean(baseline_values > threshold))
                if baseline_values.size and np.isfinite(threshold)
                else np.nan
            )
            peak_response = (
                float(np.nanmax(response_signal)) if response_signal.size else np.nan
            )

            rows.append(
                {
                    **group_info,
                    "roi": roi,
                    "stim_amp_label": amplitude_label(group_info["stim_amp_uA"]),
                    "baseline_median": baseline_median,
                    "baseline_sigma": baseline_sigma,
                    "threshold": threshold,
                    "baseline_above_threshold_fraction": baseline_above_fraction,
                    "peak_response_motion_energy_per_pixel": peak_response,
                    "sample_period_s": sample_period_s,
                    "min_frames": min_frames,
                    "min_duration_ms": params.min_duration_ms,
                    "onset_time_s": onset_time_s,
                    "latency_s": latency_s,
                    "latency_ms": latency_s * 1000.0 if np.isfinite(latency_s) else np.nan,
                    "detected": bool(np.isfinite(latency_s)),
                    "baseline_window_start_s": params.baseline_start_s,
                    "baseline_window_end_s": params.baseline_end_s,
                    "response_window_start_s": params.stim_onset_s,
                    "response_window_end_s": params.response_end_s,
                    "threshold_k": params.threshold_k,
                }
            )

    return pd.DataFrame(rows)


def summarize_latency(latency_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        latency_df.groupby(
            ["session_name", "stim_amp_uA", "stim_amp_label", "video_dir_name", "roi"],
            as_index=False,
        )
        .agg(
            n_trials=("video", "nunique"),
            n_detected=("detected", "sum"),
            detection_rate=("detected", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            std_latency_ms=("latency_ms", "std"),
            median_latency_ms=("latency_ms", "median"),
        )
        .sort_values(["session_name", "roi", "stim_amp_uA"])
    )
    return summary


def add_mean_std_overlay(ax: plt.Axes, plot_df: pd.DataFrame, order: list[str]) -> None:
    for x_pos, label in enumerate(order):
        values = plot_df.loc[plot_df["stim_amp_label"] == label, "latency_ms"].dropna()
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
            markersize=18,
            capsize=5,
            linewidth=1.5,
        )


def plot_latency_by_amplitude(
    latency_df: pd.DataFrame, session_name: str, output_dir: Path
) -> None:
    detected_df = latency_df[latency_df["detected"]].copy()
    rois = list(latency_df["roi"].drop_duplicates())
    amp_order = sorted(latency_df["stim_amp_uA"].dropna().unique())
    amp_label_order = [amplitude_label(amp) for amp in amp_order]

    fig, axes = plt.subplots(1, len(rois), figsize=(5 * len(rois), 4), squeeze=False)
    for ax, roi in zip(axes[0], rois):
        roi_df = detected_df[detected_df["roi"] == roi]
        sns.stripplot(
            data=roi_df,
            x="stim_amp_label",
            y="latency_ms",
            order=amp_label_order,
            jitter=0.22,
            size=4,
            ax=ax,
        )
        add_mean_std_overlay(ax, roi_df, amp_label_order)
        ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(amp_label_order)))
        ax.set_xticklabels(amp_label_order, rotation=30)
        ax.set_title(roi)
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Latency from stim onset (ms)")

    fig.suptitle(f"{session_name}: movement onset latency", y=1.03)
    fig.tight_layout()
    fig.savefig(output_dir / f"{session_name}_roi_latency_by_amplitude.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{session_name}_roi_latency_by_amplitude.svg", bbox_inches="tight")
    plt.close(fig)


def plot_detection_rate(
    summary_df: pd.DataFrame, session_name: str, output_dir: Path
) -> None:
    amp_order = sorted(summary_df["stim_amp_uA"].dropna().unique())
    amp_label_order = [amplitude_label(amp) for amp in amp_order]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=summary_df,
        x="stim_amp_label",
        y="detection_rate",
        hue="roi",
        style="roi",
        markers=True,
        dashes=False,
        sort=False,
        ax=ax,
    )
    ax.set_xticks(range(len(amp_label_order)))
    ax.set_xticklabels(amp_label_order, rotation=30)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{session_name}: response detection rate")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Fraction of trials detected")
    fig.tight_layout()
    fig.savefig(output_dir / f"{session_name}_roi_latency_detection_rate.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{session_name}_roi_latency_detection_rate.svg", bbox_inches="tight")
    plt.close(fig)


def session_energy_csv(session_dir: Path) -> Path:
    return (
        session_dir
        / "all_amplitudes"
        / f"{session_dir.name}_all_amplitudes_roi_motion_energy_all_frames.csv"
    )


def iter_session_dirs(motion_root: Path, sessions: list[str] | None) -> list[Path]:
    if sessions:
        return [motion_root / session for session in sessions]
    return sorted(path for path in motion_root.iterdir() if session_energy_csv(path).exists())


def analyze_session(
    session_dir: Path, output_root: Path, params: LatencyParams
) -> pd.DataFrame:
    energy_csv = session_energy_csv(session_dir)
    if not energy_csv.exists():
        raise FileNotFoundError(f"Missing motion-energy CSV: {energy_csv}")

    energy_df = pd.read_csv(energy_csv)
    rois = roi_names_from_columns(list(energy_df.columns))
    if not rois:
        raise ValueError(f"No ROI motion-energy columns found in {energy_csv}")

    session_output = output_root / session_dir.name
    session_output.mkdir(parents=True, exist_ok=True)

    latency_df = compute_latency_table(energy_df, rois, params)
    summary_df = summarize_latency(latency_df)

    latency_df.to_csv(session_output / f"{session_dir.name}_roi_motion_latency.csv", index=False)
    summary_df.to_csv(session_output / f"{session_dir.name}_roi_motion_latency_summary.csv", index=False)
    plot_latency_by_amplitude(latency_df, session_dir.name, session_output)
    plot_detection_rate(summary_df, session_dir.name, session_output)
    return latency_df


def main() -> int:
    args = parse_args()
    motion_root = args.motion_root or (
        PROJECT_ROOT / "analysis" / "video_motion_energy" / args.animal_id
    )
    output_root = args.output_root or (
        PROJECT_ROOT / "analysis" / "video_motion_latency" / args.animal_id
    )
    params = LatencyParams(
        stim_onset_s=args.stim_onset_s,
        response_end_s=args.response_end_s,
        baseline_start_s=args.baseline_start_s,
        baseline_end_s=args.baseline_end_s,
        threshold_k=args.threshold_k,
        min_duration_ms=args.min_duration_ms,
    )

    session_dirs = iter_session_dirs(motion_root, args.session)
    if not session_dirs:
        print(f"No sessions found in {motion_root}", file=sys.stderr)
        return 2

    all_latency = []
    for session_dir in session_dirs:
        latency_df = analyze_session(session_dir, output_root, params)
        all_latency.append(latency_df)
        print(f"Wrote latency analysis for {session_dir.name}")

    if len(all_latency) > 1:
        combined_dir = output_root / "all_sessions"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_df = pd.concat(all_latency, ignore_index=True)
        combined_df.to_csv(
            combined_dir / f"{args.animal_id}_roi_motion_latency_all_sessions.csv",
            index=False,
        )
        summarize_latency(combined_df).to_csv(
            combined_dir / f"{args.animal_id}_roi_motion_latency_summary_all_sessions.csv",
            index=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
