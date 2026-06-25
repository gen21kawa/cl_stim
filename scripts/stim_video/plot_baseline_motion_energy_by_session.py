"""Compare baseline ROI motion-energy distributions across sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANIMAL_ID = "M114"
DEFAULT_SESSIONS = ("M114_2026_06_17_20_10", "M114_2026_06_17_20_30")
BASELINE_PERIODS = ("pre_1", "pre_2", "pre_3")
ROI_ORDER = ("face", "forelimb")
SESSION_COLORS = {
    "M114_2026_06_17_20_10": "#2f6f9f",
    "M114_2026_06_17_20_30": "#c44e52",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot baseline pre_1/pre_2/pre_3 motion-energy distributions by session."
    )
    parser.add_argument("--animal-id", default=DEFAULT_ANIMAL_ID)
    parser.add_argument(
        "--distribution-root",
        type=Path,
        default=None,
        help="Folder containing full-video motion distribution outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for comparison plot and summary CSV.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Session ID to include. May be passed more than once.",
    )
    return parser.parse_args()


def session_label(session_id: str) -> str:
    parts = session_id.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else session_id


def load_baseline_frames(distribution_root: Path, sessions: list[str]) -> pd.DataFrame:
    frames = []
    usecols = [
        "session_id",
        "event_number",
        "roi",
        "period",
        "motion_energy_per_pixel",
    ]
    for session_id in sessions:
        csv_path = (
            distribution_root
            / session_id
            / f"{session_id}_roi_motion_energy_full_video_per_frame.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing per-frame motion-energy CSV: {csv_path}")
        session_df = pd.read_csv(csv_path, usecols=usecols)
        session_df = session_df[session_df["period"].isin(BASELINE_PERIODS)].copy()
        session_df["session_label"] = session_df["session_id"].map(session_label)
        frames.append(session_df)
    if not frames:
        raise ValueError("No sessions provided.")
    return pd.concat(frames, ignore_index=True)


def summarize_baseline(baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (session_id, session_short, roi), group in baseline_df.groupby(
        ["session_id", "session_label", "roi"], sort=False
    ):
        values = group["motion_energy_per_pixel"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "session_id": session_id,
                "session_label": session_short,
                "roi": roi,
                "periods_combined": ",".join(BASELINE_PERIODS),
                "n_trials": int(group["event_number"].nunique()),
                "n_frames": int(len(values)),
                "mean_motion_energy_per_pixel": float(np.nanmean(values)),
                "std_motion_energy_per_pixel": float(np.nanstd(values, ddof=1)),
                "median_motion_energy_per_pixel": float(np.nanmedian(values)),
                "q25_motion_energy_per_pixel": float(np.nanpercentile(values, 25)),
                "q75_motion_energy_per_pixel": float(np.nanpercentile(values, 75)),
                "iqr_motion_energy_per_pixel": float(
                    np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["roi", "session_id"])


def kde_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    values = values[np.isfinite(values)]
    if values.size < 2 or np.nanstd(values) == 0:
        return None
    try:
        return stats.gaussian_kde(values)(grid)
    except (np.linalg.LinAlgError, ValueError):
        return None


def plot_baseline_distributions(
    baseline_df: pd.DataFrame,
    summary: pd.DataFrame,
    sessions: list[str],
    output_path: Path,
) -> None:
    rois = [roi for roi in ROI_ORDER if roi in set(baseline_df["roi"])]
    rois += [roi for roi in baseline_df["roi"].drop_duplicates() if roi not in rois]

    fig, axes = plt.subplots(1, len(rois), figsize=(5.2 * len(rois), 4.0), squeeze=False)
    for ax, roi in zip(axes[0], rois):
        roi_df = baseline_df[baseline_df["roi"] == roi]
        x_max = float(np.nanpercentile(roi_df["motion_energy_per_pixel"], 99.5))
        if not np.isfinite(x_max) or x_max <= 0:
            x_max = float(np.nanmax(roi_df["motion_energy_per_pixel"]))
        grid = np.linspace(0, x_max, 350)

        for session_id in sessions:
            values = roi_df.loc[
                roi_df["session_id"].eq(session_id),
                "motion_energy_per_pixel",
            ].to_numpy(dtype=float)
            density = kde_density(values, grid)
            if density is None:
                continue
            color = SESSION_COLORS.get(session_id, None)
            row = summary[
                (summary["session_id"] == session_id) & (summary["roi"] == roi)
            ].iloc[0]
            label = (
                f"{session_label(session_id)} "
                f"(n={int(row['n_trials'])}, median={row['median_motion_energy_per_pixel']:.3f})"
            )
            ax.plot(grid, density, color=color, lw=2, label=label)
            ax.fill_between(grid, 0, density, color=color, alpha=0.18)
            ax.axvline(
                float(row["median_motion_energy_per_pixel"]),
                color=color,
                linestyle=":",
                linewidth=1.4,
            )

        ax.set_title(roi)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Baseline motion energy per pixel")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    fig.suptitle("Baseline ROI motion energy: pre_1/pre_2/pre_3 combined", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sessions = args.session or list(DEFAULT_SESSIONS)
    distribution_root = args.distribution_root or (
        PROJECT_ROOT / "analysis" / "video_motion_distribution" / args.animal_id
    )
    output_dir = args.output_dir or (distribution_root / "baseline_session_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_baseline_frames(distribution_root, sessions)
    summary = summarize_baseline(baseline_df)

    stem = "_vs_".join(session_label(session_id) for session_id in sessions)
    summary_path = output_dir / f"{args.animal_id}_{stem}_baseline_pre1_pre2_pre3_summary.csv"
    plot_path = output_dir / f"{args.animal_id}_{stem}_baseline_pre1_pre2_pre3_distribution"

    summary.to_csv(summary_path, index=False)
    plot_baseline_distributions(baseline_df, summary, sessions, plot_path)

    print(summary.to_string(index=False))
    print(f"Wrote {plot_path.with_suffix('.png')}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
