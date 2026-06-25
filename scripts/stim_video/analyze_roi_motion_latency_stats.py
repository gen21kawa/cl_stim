"""Quick exploratory statistics for ROI movement-onset latency results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANIMAL_ID = "M114"


@dataclass(frozen=True)
class GMM1DResult:
    n_components: int
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    log_likelihood: float
    bic: float
    responsibilities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory group statistics on ROI motion latency CSVs."
    )
    parser.add_argument(
        "--animal-id",
        default=DEFAULT_ANIMAL_ID,
        help=f"Animal ID. Defaults to {DEFAULT_ANIMAL_ID}.",
    )
    parser.add_argument(
        "--latency-root",
        type=Path,
        default=None,
        help="Folder containing per-session latency outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for stats CSVs and plots.",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Session ID to include. May be passed more than once.",
    )
    parser.add_argument(
        "--include-sham-in-gmm",
        action="store_true",
        help="Include sham detections when fitting the two-latency-mode model.",
    )
    return parser.parse_args()


def amplitude_label(stim_amp_uA: int | float) -> str:
    amp = int(stim_amp_uA)
    return "sham" if amp == 0 else f"{amp} uA"


def load_latency_tables(latency_root: Path, sessions: list[str] | None) -> pd.DataFrame:
    paths: list[Path] = []
    if sessions:
        for session in sessions:
            paths.append(latency_root / session / f"{session}_roi_motion_latency.csv")
    else:
        paths = sorted(latency_root.glob("M*/M*_roi_motion_latency.csv"))

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing latency CSVs:\n" + "\n".join(str(p) for p in missing))
    if not paths:
        raise FileNotFoundError(f"No per-session latency CSVs found in {latency_root}")

    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df["stim_amp_uA"] = pd.to_numeric(df["stim_amp_uA"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["detected"] = df["detected"].astype(bool)
    df["stim_amp_label"] = df["stim_amp_uA"].map(amplitude_label)
    df["is_sham"] = df["stim_amp_uA"].eq(0)
    return df


def detected_latency(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["detected"] & np.isfinite(df["latency_ms"])].copy()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(adjusted, index=p_values.index)

    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(p[finite])]
    ranked_p = p[order]
    m = len(ranked_p)
    ranked_adj = ranked_p * m / np.arange(1, m + 1)
    ranked_adj = np.minimum.accumulate(ranked_adj[::-1])[::-1]
    ranked_adj = np.clip(ranked_adj, 0.0, 1.0)
    adjusted[order] = ranked_adj
    return pd.Series(adjusted, index=p_values.index)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    comparisons = np.subtract.outer(a, b)
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["scope", "session_name", "roi", "stim_amp_uA", "stim_amp_label", "is_sham"]
    for group_values, group_df in df.groupby(group_cols, dropna=False):
        group_info = dict(zip(group_cols, group_values))
        detected = group_df[group_df["detected"] & np.isfinite(group_df["latency_ms"])]
        values = detected["latency_ms"].to_numpy(dtype=float)
        rows.append(
            {
                **group_info,
                "n_trials": int(group_df["video"].nunique()),
                "n_detected": int(detected["video"].nunique()),
                "detection_rate": float(group_df["detected"].mean()),
                "mean_latency_ms": float(np.nanmean(values)) if values.size else np.nan,
                "std_latency_ms": float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan,
                "sem_latency_ms": (
                    float(stats.sem(values, nan_policy="omit")) if values.size > 1 else np.nan
                ),
                "median_latency_ms": float(np.nanmedian(values)) if values.size else np.nan,
                "q25_latency_ms": float(np.nanpercentile(values, 25)) if values.size else np.nan,
                "q75_latency_ms": float(np.nanpercentile(values, 75)) if values.size else np.nan,
                "min_latency_ms": float(np.nanmin(values)) if values.size else np.nan,
                "max_latency_ms": float(np.nanmax(values)) if values.size else np.nan,
                "fast_latency_fraction_le_300ms": (
                    float(np.mean(values <= 300.0)) if values.size else np.nan
                ),
                "very_fast_latency_fraction_le_150ms": (
                    float(np.mean(values <= 150.0)) if values.size else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "session_name", "roi", "stim_amp_uA"])


def kruskal_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, session_name, roi), roi_df in detected_latency(df).groupby(
        ["scope", "session_name", "roi"], dropna=False
    ):
        groups = [
            group["latency_ms"].to_numpy(dtype=float)
            for _, group in roi_df.groupby("stim_amp_uA", dropna=False)
            if len(group) >= 2
        ]
        if len(groups) < 2:
            stat, p_value = np.nan, np.nan
        else:
            stat, p_value = stats.kruskal(*groups, nan_policy="omit")
        rows.append(
            {
                "scope": scope,
                "session_name": session_name,
                "roi": roi,
                "n_groups_with_at_least_2_detected_trials": len(groups),
                "kruskal_h": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pairwise_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, session_name, roi), roi_df in detected_latency(df).groupby(
        ["scope", "session_name", "roi"], dropna=False
    ):
        amp_order = sorted(roi_df["stim_amp_uA"].dropna().unique())
        for i, amp_a in enumerate(amp_order):
            for amp_b in amp_order[i + 1 :]:
                a = roi_df.loc[roi_df["stim_amp_uA"].eq(amp_a), "latency_ms"].to_numpy(dtype=float)
                b = roi_df.loc[roi_df["stim_amp_uA"].eq(amp_b), "latency_ms"].to_numpy(dtype=float)
                if len(a) < 2 or len(b) < 2:
                    stat, p_value = np.nan, np.nan
                else:
                    stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
                rows.append(
                    {
                        "scope": scope,
                        "session_name": session_name,
                        "roi": roi,
                        "amp_a_uA": amp_a,
                        "amp_b_uA": amp_b,
                        "amp_a_label": amplitude_label(amp_a),
                        "amp_b_label": amplitude_label(amp_b),
                        "n_a": len(a),
                        "n_b": len(b),
                        "median_a_ms": float(np.nanmedian(a)) if len(a) else np.nan,
                        "median_b_ms": float(np.nanmedian(b)) if len(b) else np.nan,
                        "median_b_minus_a_ms": (
                            float(np.nanmedian(b) - np.nanmedian(a))
                            if len(a) and len(b)
                            else np.nan
                        ),
                        "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        "cliffs_delta_a_vs_b": cliffs_delta(a, b),
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["p_fdr_bh"] = np.nan
    for group_values, idx in result.groupby(["scope", "session_name", "roi"]).groups.items():
        del group_values
        result.loc[idx, "p_fdr_bh"] = benjamini_hochberg(result.loc[idx, "p_value"]).to_numpy()
    return result.sort_values(["scope", "session_name", "roi", "amp_a_uA", "amp_b_uA"])


def spearman_trends(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stim_df = detected_latency(df)
    stim_df = stim_df[~stim_df["stim_amp_uA"].eq(0)]
    for (scope, session_name, roi), roi_df in stim_df.groupby(
        ["scope", "session_name", "roi"], dropna=False
    ):
        if roi_df["stim_amp_uA"].nunique() < 2 or len(roi_df) < 4:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = stats.spearmanr(roi_df["stim_amp_uA"], roi_df["latency_ms"])
        rows.append(
            {
                "scope": scope,
                "session_name": session_name,
                "roi": roi,
                "n_detected_stim_trials": int(len(roi_df)),
                "n_amplitudes": int(roi_df["stim_amp_uA"].nunique()),
                "spearman_rho_amplitude_vs_latency": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def normal_pdf(values: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    safe_var = np.maximum(variances, 1e-6)
    scale = np.sqrt(2.0 * np.pi * safe_var)
    exponent = -0.5 * ((values[:, None] - means[None, :]) ** 2) / safe_var[None, :]
    return np.exp(exponent) / scale[None, :]


def gmm_1d(values: np.ndarray, n_components: int, max_iter: int = 500) -> GMM1DResult:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        nan_array = np.full(n_components, np.nan)
        return GMM1DResult(n_components, nan_array, nan_array, nan_array, np.nan, np.nan, np.empty((0, n_components)))

    if n_components == 1:
        mean = np.array([float(np.mean(values))])
        variance = np.array([float(np.var(values) + 1e-6)])
        weights = np.array([1.0])
        likelihood = np.sum(np.log(normal_pdf(values, mean, variance).ravel() + 1e-300))
        bic = (2 * np.log(n)) - (2 * likelihood)
        return GMM1DResult(1, mean, variance, weights, float(likelihood), float(bic), np.ones((n, 1)))

    quantiles = np.linspace(0, 100, n_components + 2)[1:-1]
    means = np.percentile(values, quantiles).astype(float)
    variances = np.full(n_components, float(np.var(values) + 1e-6))
    weights = np.full(n_components, 1.0 / n_components)
    last_log_likelihood = -np.inf

    for _ in range(max_iter):
        weighted_pdf = normal_pdf(values, means, variances) * weights[None, :]
        total_pdf = np.sum(weighted_pdf, axis=1, keepdims=True) + 1e-300
        responsibilities = weighted_pdf / total_pdf
        effective_n = responsibilities.sum(axis=0) + 1e-300

        weights = effective_n / n
        means = (responsibilities * values[:, None]).sum(axis=0) / effective_n
        variances = (
            responsibilities * ((values[:, None] - means[None, :]) ** 2)
        ).sum(axis=0) / effective_n
        variances = np.maximum(variances, 1e-6)

        weighted_pdf = normal_pdf(values, means, variances) * weights[None, :]
        log_likelihood = float(np.sum(np.log(np.sum(weighted_pdf, axis=1) + 1e-300)))
        if abs(log_likelihood - last_log_likelihood) < 1e-6:
            break
        last_log_likelihood = log_likelihood

    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    weighted_pdf = normal_pdf(values, means, variances) * weights[None, :]
    responsibilities = weighted_pdf / (np.sum(weighted_pdf, axis=1, keepdims=True) + 1e-300)
    log_likelihood = float(np.sum(np.log(np.sum(weighted_pdf, axis=1) + 1e-300)))
    n_params = (3 * n_components) - 1
    bic = float((n_params * np.log(n)) - (2 * log_likelihood))
    return GMM1DResult(n_components, means, variances, weights, log_likelihood, bic, responsibilities)


def two_component_latency_modes(
    df: pd.DataFrame, include_sham: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_rows = []
    assignment_frames = []
    by_amp_rows = []
    candidate_df = detected_latency(df)
    if not include_sham:
        candidate_df = candidate_df[~candidate_df["stim_amp_uA"].eq(0)]

    for (scope, session_name, roi), roi_df in candidate_df.groupby(
        ["scope", "session_name", "roi"], dropna=False
    ):
        roi_df = roi_df.sort_values(["stim_amp_uA", "video"]).copy()
        values = roi_df["latency_ms"].to_numpy(dtype=float)
        if len(values) < 6:
            continue

        one = gmm_1d(values, 1)
        two = gmm_1d(values, 2)
        responsibilities = two.responsibilities
        early_prob = responsibilities[:, 0]
        component = np.where(early_prob >= 0.5, "early", "delayed")

        assigned = roi_df.copy()
        assigned["latency_mode"] = component
        assigned["early_mode_probability"] = early_prob
        assignment_frames.append(assigned)

        bic_delta = one.bic - two.bic
        mode_rows.append(
            {
                "scope": scope,
                "session_name": session_name,
                "roi": roi,
                "n_detected_trials_used": int(len(values)),
                "include_sham": include_sham,
                "one_component_bic": one.bic,
                "two_component_bic": two.bic,
                "bic_improvement_two_component": bic_delta,
                "prefers_two_components": bool(bic_delta > 0),
                "early_mean_ms": two.means[0],
                "delayed_mean_ms": two.means[1],
                "early_sd_ms": float(np.sqrt(two.variances[0])),
                "delayed_sd_ms": float(np.sqrt(two.variances[1])),
                "early_weight": two.weights[0],
                "delayed_weight": two.weights[1],
            }
        )

        for (amp, label), amp_df in assigned.groupby(["stim_amp_uA", "stim_amp_label"]):
            n_detected = len(amp_df)
            n_early = int((amp_df["latency_mode"] == "early").sum())
            by_amp_rows.append(
                {
                    "scope": scope,
                    "session_name": session_name,
                    "roi": roi,
                    "stim_amp_uA": amp,
                    "stim_amp_label": label,
                    "n_detected": n_detected,
                    "n_early_mode": n_early,
                    "n_delayed_mode": int(n_detected - n_early),
                    "early_mode_fraction": float(n_early / n_detected) if n_detected else np.nan,
                    "median_latency_ms": float(amp_df["latency_ms"].median()),
                }
            )

    modes = pd.DataFrame(mode_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    by_amp = pd.DataFrame(by_amp_rows)
    return modes, assignments, by_amp


def add_all_sessions_scope(df: pd.DataFrame) -> pd.DataFrame:
    per_session = df.copy()
    per_session["scope"] = "per_session"

    pooled = df.copy()
    pooled["scope"] = "all_sessions"
    pooled["session_name"] = "all_sessions"
    return pd.concat([per_session, pooled], ignore_index=True)


def plot_pairwise_heatmaps(pairwise: pd.DataFrame, output_dir: Path) -> None:
    if pairwise.empty:
        return

    for (scope, session_name), plot_df in pairwise.groupby(["scope", "session_name"]):
        rois = list(plot_df["roi"].drop_duplicates())
        fig, axes = plt.subplots(
            1,
            len(rois),
            figsize=(5 * len(rois), 4),
            squeeze=False,
            constrained_layout=True,
        )
        images = []
        for ax, roi in zip(axes[0], rois):
            roi_df = plot_df[plot_df["roi"] == roi]
            amps = sorted(set(roi_df["amp_a_uA"]).union(set(roi_df["amp_b_uA"])))
            labels = [amplitude_label(amp) for amp in amps]
            matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
            text = pd.DataFrame("", index=labels, columns=labels)
            for _, row in roi_df.iterrows():
                a = amplitude_label(row["amp_a_uA"])
                b = amplitude_label(row["amp_b_uA"])
                p_fdr = row["p_fdr_bh"]
                if np.isfinite(p_fdr):
                    neg_log_p = -np.log10(max(float(p_fdr), 1e-300))
                    matrix.loc[a, b] = neg_log_p
                    text.loc[a, b] = f"{p_fdr:.3g}"
            masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
            im = ax.imshow(masked, cmap="magma", aspect="auto")
            images.append(im)
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    label = text.iloc[y, x]
                    if label:
                        value = matrix.iloc[y, x]
                        color = "white" if value > 1.3 else "black"
                        ax.text(x, y, label, ha="center", va="center", fontsize=7, color=color)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_title(roi)
            ax.set_xlabel("Higher amplitude group")
            ax.set_ylabel("Lower amplitude group")
        if images:
            fig.colorbar(
                images[-1],
                ax=axes.ravel().tolist(),
                label="-log10(FDR p)",
                shrink=0.8,
                pad=0.02,
            )
        fig.suptitle(f"{session_name}: pairwise Mann-Whitney FDR p-values")
        safe_name = f"{session_name}_latency_pairwise_mannwhitney_fdr_heatmap"
        fig.savefig(output_dir / f"{safe_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{safe_name}.svg", bbox_inches="tight")
        plt.close(fig)


def plot_gmm_modes(assignments: pd.DataFrame, modes: pd.DataFrame, output_dir: Path) -> None:
    if assignments.empty or modes.empty:
        return

    palette = {"early": "#1f77b4", "delayed": "#d62728"}
    rng = np.random.default_rng(20260619)
    for (scope, session_name), plot_df in assignments.groupby(["scope", "session_name"]):
        rois = list(plot_df["roi"].drop_duplicates())
        fig, axes = plt.subplots(1, len(rois), figsize=(5 * len(rois), 4), squeeze=False)
        for ax, roi in zip(axes[0], rois):
            roi_df = plot_df[plot_df["roi"] == roi].copy()
            amp_order = sorted(roi_df["stim_amp_uA"].dropna().unique())
            label_order = [amplitude_label(amp) for amp in amp_order]
            x_lookup = {label: idx for idx, label in enumerate(label_order)}
            for mode, mode_df in roi_df.groupby("latency_mode"):
                x = mode_df["stim_amp_label"].map(x_lookup).to_numpy(dtype=float)
                jitter = rng.uniform(-0.22, 0.22, size=len(mode_df))
                ax.scatter(
                    x + jitter,
                    mode_df["latency_ms"],
                    color=palette.get(mode, "0.5"),
                    label=mode,
                    s=18,
                    alpha=0.85,
                )
            mode_row = modes[
                (modes["scope"] == scope)
                & (modes["session_name"] == session_name)
                & (modes["roi"] == roi)
            ]
            if not mode_row.empty:
                early = float(mode_row.iloc[0]["early_mean_ms"])
                delayed = float(mode_row.iloc[0]["delayed_mean_ms"])
                ax.axhline(early, color=palette["early"], linestyle=":", linewidth=1)
                ax.axhline(delayed, color=palette["delayed"], linestyle=":", linewidth=1)
            ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
            ax.set_xticks(range(len(label_order)))
            ax.set_xticklabels(label_order, rotation=30)
            ax.set_title(roi)
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Latency from stim onset (ms)")
        handles, labels = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, labels, title="GMM mode", loc="upper right")
        fig.suptitle(f"{session_name}: two-latency-mode exploratory fit", y=1.03)
        fig.tight_layout()
        safe_name = f"{session_name}_latency_two_component_gmm_modes"
        fig.savefig(output_dir / f"{safe_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{safe_name}.svg", bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    latency_root = args.latency_root or (
        PROJECT_ROOT / "analysis" / "video_motion_latency" / args.animal_id
    )
    output_dir = args.output_dir or (latency_root / "stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    latency_df = load_latency_tables(latency_root, args.session)
    scoped_df = add_all_sessions_scope(latency_df)

    group_summary = summarize_groups(scoped_df)
    kruskal = kruskal_tests(scoped_df)
    pairwise = pairwise_mannwhitney(scoped_df)
    trends = spearman_trends(scoped_df)
    modes, assignments, mode_by_amp = two_component_latency_modes(
        scoped_df, include_sham=args.include_sham_in_gmm
    )

    group_summary.to_csv(output_dir / f"{args.animal_id}_latency_group_summary.csv", index=False)
    kruskal.to_csv(output_dir / f"{args.animal_id}_latency_kruskal_by_roi.csv", index=False)
    pairwise.to_csv(output_dir / f"{args.animal_id}_latency_pairwise_mannwhitney.csv", index=False)
    trends.to_csv(output_dir / f"{args.animal_id}_latency_spearman_amplitude_trends.csv", index=False)
    modes.to_csv(output_dir / f"{args.animal_id}_latency_two_component_gmm_summary.csv", index=False)
    assignments.to_csv(output_dir / f"{args.animal_id}_latency_two_component_gmm_assignments.csv", index=False)
    mode_by_amp.to_csv(output_dir / f"{args.animal_id}_latency_gmm_mode_by_amplitude.csv", index=False)

    plot_pairwise_heatmaps(pairwise, output_dir)
    plot_gmm_modes(assignments, modes, output_dir)

    print(f"Wrote latency stats to {output_dir}")
    print(
        group_summary[
            ["scope", "session_name", "roi", "stim_amp_label", "n_detected", "median_latency_ms"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
