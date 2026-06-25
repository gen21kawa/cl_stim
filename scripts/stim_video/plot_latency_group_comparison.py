"""Plot pooled latency group comparisons for one stimulation session."""

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
DEFAULT_SESSION = "M114_2026_06_17_20_10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pooled intermediate-amplitude latencies against a high-amplitude group."
    )
    parser.add_argument("--animal-id", default=DEFAULT_ANIMAL_ID)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--latency-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--pooled-amps",
        type=float,
        nargs="+",
        default=[25.0, 50.0, 75.0],
        help="Amplitudes to pool into the intermediate group.",
    )
    parser.add_argument("--comparison-amp", type=float, default=100.0)
    return parser.parse_args()


def amplitude_text(amplitudes: list[float]) -> str:
    amps = [int(amp) if float(amp).is_integer() else amp for amp in amplitudes]
    if len(amps) == 1:
        return f"{amps[0]} uA"
    return f"{amps[0]}-{amps[-1]} uA"


def p_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    comparisons = np.subtract.outer(a, b)
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def deterministic_swarm_offsets(values: np.ndarray, max_width: float = 0.22) -> np.ndarray:
    """Return stable offsets that spread nearby y-values within each category."""
    if len(values) == 0:
        return np.array([], dtype=float)

    order = np.argsort(values)
    offsets = np.zeros(len(values), dtype=float)
    y_range = float(np.nanmax(values) - np.nanmin(values)) if len(values) > 1 else 1.0
    bin_width = max(y_range * 0.055, 35.0)
    bins: dict[int, list[int]] = {}
    y_min = float(np.nanmin(values))

    for idx in order:
        bin_idx = int(np.floor((values[idx] - y_min) / bin_width))
        nearby_count = sum(len(bins.get(bin_idx + delta, [])) for delta in (-1, 0, 1))
        direction = -1 if nearby_count % 2 else 1
        magnitude = ((nearby_count + 1) // 2) * (max_width / 4.0)
        offsets[idx] = float(np.clip(direction * magnitude, -max_width, max_width))
        bins.setdefault(bin_idx, []).append(idx)
    return offsets


def load_latency_data(latency_root: Path, session: str) -> pd.DataFrame:
    csv_path = latency_root / session / f"{session}_roi_motion_latency.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing latency CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df["stim_amp_uA"] = pd.to_numeric(df["stim_amp_uA"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["detected"] = df["detected"].astype(bool)
    return df


def build_comparison_table(
    df: pd.DataFrame,
    pooled_amps: list[float],
    comparison_amp: float,
    pooled_label: str,
    comparison_label: str,
) -> pd.DataFrame:
    rows = []
    detected = df[df["detected"] & np.isfinite(df["latency_ms"])].copy()

    for roi, roi_df in detected.groupby("roi", sort=False):
        pooled = roi_df.loc[roi_df["stim_amp_uA"].isin(pooled_amps), "latency_ms"].to_numpy(float)
        comparison = roi_df.loc[
            roi_df["stim_amp_uA"].eq(comparison_amp), "latency_ms"
        ].to_numpy(float)
        if len(pooled) >= 1 and len(comparison) >= 1:
            u_stat, p_value = stats.mannwhitneyu(pooled, comparison, alternative="two-sided")
        else:
            u_stat, p_value = np.nan, np.nan

        rows.append(
            {
                "roi": roi,
                "group_a": pooled_label,
                "group_b": comparison_label,
                "group_a_amplitudes_uA": ",".join(str(int(amp)) for amp in pooled_amps),
                "group_b_amplitude_uA": comparison_amp,
                "n_group_a": len(pooled),
                "n_group_b": len(comparison),
                "median_group_a_ms": float(np.nanmedian(pooled)) if len(pooled) else np.nan,
                "q25_group_a_ms": float(np.nanpercentile(pooled, 25)) if len(pooled) else np.nan,
                "q75_group_a_ms": float(np.nanpercentile(pooled, 75)) if len(pooled) else np.nan,
                "median_group_b_ms": (
                    float(np.nanmedian(comparison)) if len(comparison) else np.nan
                ),
                "q25_group_b_ms": (
                    float(np.nanpercentile(comparison, 25)) if len(comparison) else np.nan
                ),
                "q75_group_b_ms": (
                    float(np.nanpercentile(comparison, 75)) if len(comparison) else np.nan
                ),
                "median_b_minus_a_ms": (
                    float(np.nanmedian(comparison) - np.nanmedian(pooled))
                    if len(pooled) and len(comparison)
                    else np.nan
                ),
                "mannwhitney_u": float(u_stat) if np.isfinite(u_stat) else np.nan,
                "p_value_two_sided": float(p_value) if np.isfinite(p_value) else np.nan,
                "cliffs_delta_group_a_vs_b": cliffs_delta(pooled, comparison),
            }
        )

    return pd.DataFrame(rows)


def plot_group_comparison(
    df: pd.DataFrame,
    comparison_table: pd.DataFrame,
    pooled_amps: list[float],
    comparison_amp: float,
    pooled_label: str,
    comparison_label: str,
    session: str,
    output_dir: Path,
) -> None:
    detected = df[df["detected"] & np.isfinite(df["latency_ms"])].copy()
    detected["latency_group"] = pd.Series(pd.NA, index=detected.index, dtype="object")
    detected.loc[detected["stim_amp_uA"].isin(pooled_amps), "latency_group"] = pooled_label
    detected.loc[detected["stim_amp_uA"].eq(comparison_amp), "latency_group"] = comparison_label
    plot_df = detected[detected["latency_group"].isin([pooled_label, comparison_label])].copy()

    rois = [roi for roi in ["face", "forelimb"] if roi in plot_df["roi"].unique()]
    rois += [roi for roi in plot_df["roi"].drop_duplicates() if roi not in rois]
    colors = {pooled_label: "#6f7f8f", comparison_label: "#d62728"}

    fig, axes = plt.subplots(1, len(rois), figsize=(5 * len(rois), 4.4), squeeze=False)
    for ax, roi in zip(axes[0], rois):
        roi_df = plot_df[plot_df["roi"] == roi]
        y_max = float(roi_df["latency_ms"].max())
        y_min = min(0.0, float(roi_df["latency_ms"].min()))
        y_pad = max((y_max - y_min) * 0.16, 180.0)

        for x_pos, label in enumerate([pooled_label, comparison_label]):
            values = roi_df.loc[roi_df["latency_group"] == label, "latency_ms"].to_numpy(float)
            offsets = deterministic_swarm_offsets(values)
            ax.scatter(
                x_pos + offsets,
                values,
                s=34,
                color=colors[label],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
            if len(values):
                median = float(np.nanmedian(values))
                q25 = float(np.nanpercentile(values, 25))
                q75 = float(np.nanpercentile(values, 75))
                ax.plot([x_pos - 0.22, x_pos + 0.22], [median, median], color="black", lw=2)
                ax.plot([x_pos, x_pos], [q25, q75], color="black", lw=1.3)
                ax.plot([x_pos - 0.08, x_pos + 0.08], [q25, q25], color="black", lw=1.3)
                ax.plot([x_pos - 0.08, x_pos + 0.08], [q75, q75], color="black", lw=1.3)

        row = comparison_table[comparison_table["roi"] == roi].iloc[0]
        p_value = float(row["p_value_two_sided"])
        bracket_y = y_max + (y_pad * 0.35)
        text_y = y_max + (y_pad * 0.55)
        ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + 35, bracket_y + 35, bracket_y], color="black", lw=1)
        ax.text(
            0.5,
            text_y,
            f"{p_to_stars(p_value)}  p={p_value:.2g}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        n_a = int(row["n_group_a"])
        n_b = int(row["n_group_b"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{pooled_label}\n(n={n_a})", f"{comparison_label}\n(n={n_b})"])
        ax.set_ylim(y_min - 50, y_max + y_pad)
        ax.axhline(0, color="0.4", linestyle=":", linewidth=1)
        ax.set_title(roi)
        ax.set_ylabel("Latency from stim onset (ms)")
        ax.set_xlabel("Stimulation amplitude group")

    fig.suptitle(f"{session}: pooled latency comparison", y=1.03)
    fig.tight_layout()
    stem = f"{session}_latency_{pooled_label.replace(' ', '_').replace('-', '_')}_vs_{comparison_label.replace(' ', '_')}"
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    latency_root = args.latency_root or (
        PROJECT_ROOT / "analysis" / "video_motion_latency" / args.animal_id
    )
    output_dir = args.output_dir or (latency_root / "stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_amps = sorted(args.pooled_amps)
    pooled_label = amplitude_text(pooled_amps)
    comparison_label = amplitude_text([args.comparison_amp])

    df = load_latency_data(latency_root, args.session)
    comparison_table = build_comparison_table(
        df,
        pooled_amps=pooled_amps,
        comparison_amp=args.comparison_amp,
        pooled_label=pooled_label,
        comparison_label=comparison_label,
    )
    stem = (
        f"{args.session}_latency_{pooled_label.replace(' ', '_').replace('-', '_')}"
        f"_vs_{comparison_label.replace(' ', '_')}"
    )
    comparison_table.to_csv(output_dir / f"{stem}_mannwhitney.csv", index=False)
    plot_group_comparison(
        df,
        comparison_table,
        pooled_amps=pooled_amps,
        comparison_amp=args.comparison_amp,
        pooled_label=pooled_label,
        comparison_label=comparison_label,
        session=args.session,
        output_dir=output_dir,
    )

    print(comparison_table.to_string(index=False))
    print(f"Wrote plot and stats to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
