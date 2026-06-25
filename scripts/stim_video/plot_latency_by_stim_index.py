"""Plot ROI movement latency against stimulation index within one amplitude."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANIMAL_ID = "M114"
DEFAULT_SESSION = "M114_2026_06_17_20_10"
DEFAULT_AMPLITUDE_UA = 75.0
ROI_ORDER = ("face", "forelimb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot detected ROI latency by within-amplitude stimulation index."
    )
    parser.add_argument("--animal-id", default=DEFAULT_ANIMAL_ID)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--amplitude-uA", type=float, default=DEFAULT_AMPLITUDE_UA)
    parser.add_argument("--latency-root", type=Path, default=None)
    parser.add_argument("--summary-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def amplitude_label(amplitude_uA: float) -> str:
    return f"{int(amplitude_uA)} uA" if float(amplitude_uA).is_integer() else f"{amplitude_uA:g} uA"


def safe_amplitude_label(amplitude_uA: float) -> str:
    return amplitude_label(amplitude_uA).replace(" ", "_")


def parse_event_number(video_name: str) -> int:
    match = re.search(r"event_(\d+)_", str(video_name))
    if match is None:
        raise ValueError(f"Could not parse event number from video name: {video_name}")
    return int(match.group(1))


def load_latency_table(latency_root: Path, session: str, amplitude_uA: float) -> pd.DataFrame:
    latency_path = latency_root / session / f"{session}_roi_motion_latency.csv"
    if not latency_path.exists():
        raise FileNotFoundError(f"Missing latency CSV: {latency_path}")
    latency = pd.read_csv(latency_path)
    latency["stim_amp_uA"] = pd.to_numeric(latency["stim_amp_uA"], errors="coerce")
    latency["latency_ms"] = pd.to_numeric(latency["latency_ms"], errors="coerce")
    latency = latency[np.isclose(latency["stim_amp_uA"], amplitude_uA)].copy()
    latency["event_number"] = latency["video"].map(parse_event_number)
    return latency


def load_stim_index(summary_root: Path, session: str, amplitude_uA: float) -> pd.DataFrame:
    summary_path = summary_root / session / f"{session}_stim_motion_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing stim summary CSV: {summary_path}")
    summary = pd.read_csv(summary_path)
    summary["amplitude_uA"] = pd.to_numeric(summary["amplitude_uA"], errors="coerce")
    summary = summary[np.isclose(summary["amplitude_uA"], amplitude_uA)].copy()
    summary = summary.sort_values("window_start_ms").reset_index(drop=True)
    summary["stim_index"] = np.arange(1, len(summary) + 1)
    return summary[["event_number", "window_start_ms", "stim_index"]]


def build_plot_table(
    latency_root: Path,
    summary_root: Path,
    session: str,
    amplitude_uA: float,
) -> pd.DataFrame:
    latency = load_latency_table(latency_root, session, amplitude_uA)
    stim_index = load_stim_index(summary_root, session, amplitude_uA)
    merged = latency.merge(stim_index, on="event_number", how="left", validate="many_to_one")
    if merged["stim_index"].isna().any():
        missing = merged.loc[merged["stim_index"].isna(), "event_number"].drop_duplicates()
        raise ValueError(f"Missing stim index for event numbers: {missing.tolist()}")
    return merged.sort_values(["stim_index", "roi"]).reset_index(drop=True)


def trend_summary(plot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    detected = plot_df[plot_df["detected"] & np.isfinite(plot_df["latency_ms"])].copy()
    for roi, roi_df in detected.groupby("roi", sort=False):
        x = roi_df["stim_index"].to_numpy(dtype=float)
        y = roi_df["latency_ms"].to_numpy(dtype=float)
        if len(roi_df) >= 3:
            spearman_rho, spearman_p = stats.spearmanr(x, y)
            slope, intercept, pearson_r, pearson_p, slope_stderr = stats.linregress(x, y)
        else:
            spearman_rho = spearman_p = slope = intercept = pearson_r = pearson_p = slope_stderr = np.nan
        rows.append(
            {
                "roi": roi,
                "n_detected": int(len(roi_df)),
                "spearman_rho_stim_index_vs_latency": float(spearman_rho)
                if np.isfinite(spearman_rho)
                else np.nan,
                "spearman_p_value": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
                "linear_slope_ms_per_stim": float(slope) if np.isfinite(slope) else np.nan,
                "linear_intercept_ms": float(intercept) if np.isfinite(intercept) else np.nan,
                "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
                "pearson_p_value": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
                "slope_stderr": float(slope_stderr) if np.isfinite(slope_stderr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_latency_by_index(
    plot_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    session: str,
    amplitude_uA: float,
    output_path: Path,
) -> None:
    rois = [roi for roi in ROI_ORDER if roi in set(plot_df["roi"])]
    rois += [roi for roi in plot_df["roi"].drop_duplicates() if roi not in rois]

    fig, axes = plt.subplots(1, len(rois), figsize=(5.4 * len(rois), 4.2), squeeze=False)
    for ax, roi in zip(axes[0], rois):
        roi_df = plot_df[plot_df["roi"] == roi].copy()
        detected = roi_df[roi_df["detected"] & np.isfinite(roi_df["latency_ms"])]
        undetected = roi_df[~roi_df["detected"] | ~np.isfinite(roi_df["latency_ms"])]

        ax.plot(
            detected["stim_index"],
            detected["latency_ms"],
            color="#4c78a8",
            linewidth=1.3,
            alpha=0.6,
        )
        ax.scatter(
            detected["stim_index"],
            detected["latency_ms"],
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.6,
            s=42,
            zorder=3,
            label="detected",
        )
        if not undetected.empty:
            ax.scatter(
                undetected["stim_index"],
                np.zeros(len(undetected)),
                marker="x",
                color="0.4",
                s=35,
                label="not detected",
            )

        trend = trend_df[trend_df["roi"] == roi]
        if not trend.empty and np.isfinite(trend.iloc[0]["linear_slope_ms_per_stim"]):
            slope = float(trend.iloc[0]["linear_slope_ms_per_stim"])
            intercept = float(trend.iloc[0]["linear_intercept_ms"])
            x_line = np.array([detected["stim_index"].min(), detected["stim_index"].max()])
            ax.plot(
                x_line,
                intercept + slope * x_line,
                color="#d62728",
                linestyle="--",
                linewidth=1.5,
                label="linear fit",
            )
            text = (
                f"Spearman rho={trend.iloc[0]['spearman_rho_stim_index_vs_latency']:.2f}\n"
                f"p={trend.iloc[0]['spearman_p_value']:.3g}\n"
                f"slope={slope:.1f} ms/stim"
            )
            ax.text(
                0.03,
                0.97,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
            )

        ax.axhline(0, color="0.5", linestyle=":", linewidth=1)
        ax.set_title(roi)
        ax.set_xlabel(f"{amplitude_label(amplitude_uA)} stimulation index")
        ax.set_ylabel("Latency from stim onset (ms)")
        ax.set_xticks(sorted(plot_df["stim_index"].dropna().unique()))
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(f"{session}: {amplitude_label(amplitude_uA)} latency over repeated stimulations", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    latency_root = args.latency_root or (
        PROJECT_ROOT / "analysis" / "video_motion_latency" / args.animal_id
    )
    summary_root = args.summary_root or (
        PROJECT_ROOT / "analysis" / "stim_motion" / args.animal_id
    )
    output_dir = args.output_dir or (latency_root / "stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = build_plot_table(
        latency_root=latency_root,
        summary_root=summary_root,
        session=args.session,
        amplitude_uA=args.amplitude_uA,
    )
    trend_df = trend_summary(plot_df)

    stem = f"{args.session}_{safe_amplitude_label(args.amplitude_uA)}_latency_by_stim_index"
    plot_df.to_csv(output_dir / f"{stem}_data.csv", index=False)
    trend_df.to_csv(output_dir / f"{stem}_trend_stats.csv", index=False)
    plot_latency_by_index(plot_df, trend_df, args.session, args.amplitude_uA, output_dir / stem)

    print(trend_df.to_string(index=False))
    print(f"Wrote {output_dir / (stem + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
