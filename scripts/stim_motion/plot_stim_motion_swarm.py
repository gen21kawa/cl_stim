"""Plot stimulation amplitude against MotSen1 motion response."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stim_motion_summary import motion_counts_to_cm, summarize_animal, write_summary_csv


DEFAULT_ANIMAL_ROOT = PROJECT_ROOT / "data" / "raw" / "M114"
PLOT_KINDS = ("swarm", "box", "violin")


@dataclass(frozen=True)
class PlotPoint:
    """One point to show in the amplitude-motion swarm plot."""

    amplitude_uA: float
    motion_distance_cm: float
    session_id: str
    event_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a swarm plot of stimulation amplitude vs motion."
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
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Existing or desired combined summary CSV path.",
    )
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Rebuild the stimulation-motion CSV before plotting.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <output-root>/plots/<animal>_amplitude_motion_swarm.png.",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="SVG output path. Defaults to <output-root>/plots/<animal>_amplitude_motion_swarm.svg.",
    )
    parser.add_argument(
        "--per-session",
        action="store_true",
        help="Also write one amplitude-motion swarm plot per session summary CSV.",
    )
    return parser.parse_args()


def default_plot_root(output_root: str | Path) -> Path:
    """Return the default folder for generated figures."""

    return Path(output_root) / "plots"


def read_summary_csv(summary_csv: str | Path) -> list[dict[str, str]]:
    with Path(summary_csv).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iter_session_summary_csvs(output_root: str | Path) -> list[Path]:
    """Return per-session summary CSVs, including legacy filenames."""

    root = Path(output_root)
    summaries: list[Path] = []
    for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if session_dir.name == "plots":
            continue
        session_named = session_dir / f"{session_dir.name}_stim_motion_summary.csv"
        legacy = session_dir / "stim_motion_summary.csv"
        if session_named.exists():
            summaries.append(session_named)
        elif legacy.exists():
            summaries.append(legacy)
    return summaries


def _as_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def prepare_plot_points(rows: list[dict[str, object]]) -> tuple[list[PlotPoint], int]:
    """Return finite matched rows and a count of omitted rows."""

    points: list[PlotPoint] = []
    omitted = 0
    for row in rows:
        if str(row.get("matched_pycontrol_window", "")).lower() != "true":
            omitted += 1
            continue
        amplitude = _as_finite_float(row.get("amplitude_uA"))
        motion = _as_finite_float(row.get("motion_distance_cm"))
        if motion is None:
            legacy_motion_counts = _as_finite_float(row.get("motion_magnitude_sum"))
            motion = (
                motion_counts_to_cm(legacy_motion_counts)
                if legacy_motion_counts is not None
                else None
            )
        if amplitude is None or motion is None:
            omitted += 1
            continue
        try:
            event_number = int(row.get("event_number", 0))
        except (TypeError, ValueError):
            event_number = 0
        points.append(
            PlotPoint(
                amplitude_uA=amplitude,
                motion_distance_cm=motion,
                session_id=str(row.get("session_id", "")),
                event_number=event_number,
            )
        )
    return points, omitted


def compute_swarm_offsets(count: int, *, max_width: float = 0.32) -> list[float]:
    """Return deterministic symmetric offsets for one amplitude group."""

    if count <= 0:
        return []
    ranks = [0]
    for index in range(1, count):
        rank = (index + 1) // 2
        ranks.append(-rank if index % 2 else rank)
    max_rank = max(abs(rank) for rank in ranks)
    if max_rank == 0:
        return [0.0]
    step = max_width / max_rank
    return [rank * step for rank in ranks]


def _build_swarm_coordinates(
    points: list[PlotPoint],
) -> tuple[list[float], list[float], list[str], list[float]]:
    amplitudes = sorted({point.amplitude_uA for point in points})
    if len(amplitudes) > 1:
        min_gap = min(
            right - left for left, right in zip(amplitudes[:-1], amplitudes[1:])
        )
        swarm_width = max(0.25, min_gap * 0.28)
    else:
        swarm_width = 1.0
    x_values: list[float] = []
    y_values: list[float] = []
    sessions: list[str] = []

    for amplitude in amplitudes:
        group = [point for point in points if point.amplitude_uA == amplitude]
        group.sort(
            key=lambda point: (
                point.motion_distance_cm,
                point.session_id,
                point.event_number,
            )
        )
        offsets = compute_swarm_offsets(len(group), max_width=swarm_width)
        for point, offset in zip(group, offsets):
            x_values.append(amplitude + offset)
            y_values.append(point.motion_distance_cm)
            sessions.append(point.session_id)

    return x_values, y_values, sessions, amplitudes


def compute_amplitude_means(points: list[PlotPoint]) -> list[tuple[float, float]]:
    """Return mean motion for each stimulation amplitude."""

    means: list[tuple[float, float]] = []
    for amplitude in sorted({point.amplitude_uA for point in points}):
        values = [
            point.motion_distance_cm
            for point in points
            if point.amplitude_uA == amplitude
        ]
        if values:
            means.append((amplitude, sum(values) / len(values)))
    return means


def _session_colors(plt, sessions: list[str]) -> dict[str, str]:
    unique_sessions = sorted(set(sessions))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    return {
        session: color_cycle[index % len(color_cycle)]
        for index, session in enumerate(unique_sessions)
    }


def _plot_session_points(
    ax,
    x_values: list[float],
    y_values: list[float],
    sessions: list[str],
    session_colors: dict[str, str],
) -> None:
    for session in sorted(session_colors):
        indices = [index for index, value in enumerate(sessions) if value == session]
        ax.scatter(
            [x_values[index] for index in indices],
            [y_values[index] for index in indices],
            s=36,
            alpha=0.78,
            color=session_colors[session],
            edgecolor="white",
            linewidth=0.5,
            label=session,
            zorder=4,
        )


def _plot_mean_markers(ax, points: list[PlotPoint]) -> None:
    amplitude_means = compute_amplitude_means(points)
    ax.scatter(
        [amplitude for amplitude, _ in amplitude_means],
        [mean for _, mean in amplitude_means],
        s=88,
        marker="D",
        color="black",
        edgecolor="white",
        linewidth=0.8,
        label="Mean",
        zorder=5,
    )


def _style_amplitude_motion_axis(
    ax,
    *,
    amplitudes: list[float],
    title_label: str,
    plot_label: str,
    plotted: int,
    omitted: int,
) -> None:
    ax.set_xticks(amplitudes)
    ax.set_xticklabels([f"{amplitude:g}" for amplitude in amplitudes])
    ax.set_xlabel("Stimulation amplitude (uA)")
    ax.set_ylabel("Motion distance (cm)")
    ax.set_title(
        f"{title_label}: stimulation amplitude vs motion ({plot_label})\n"
        f"{plotted} matched stimulations plotted; {omitted} unmatched/NaN rows omitted"
    )
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _amplitude_group_values(
    points: list[PlotPoint],
    amplitudes: list[float],
) -> list[list[float]]:
    return [
        [point.motion_distance_cm for point in points if point.amplitude_uA == amplitude]
        for amplitude in amplitudes
    ]


def _violin_group_values(
    points: list[PlotPoint],
    amplitudes: list[float],
) -> list[list[float]]:
    groups = _amplitude_group_values(points, amplitudes)
    stable_groups: list[list[float]] = []
    for values in groups:
        if len(values) == 1:
            value = values[0]
            epsilon = max(abs(value) * 1e-9, 1e-9)
            stable_groups.append([value - epsilon, value + epsilon])
        elif len(set(values)) == 1:
            value = values[0]
            epsilon = max(abs(value) * 1e-9, 1e-9)
            stable_groups.append([value - epsilon, value + epsilon, *values])
        else:
            stable_groups.append(values)
    return stable_groups


def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "cl_stim_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def plot_amplitude_motion_from_summary(
    summary_csv: str | Path,
    *,
    png_path: str | Path,
    svg_path: str | Path,
    animal_id: str,
    plot_kind: str,
) -> tuple[Path, Path, int, int]:
    """Create PNG and SVG amplitude-motion distribution plots."""

    if plot_kind not in PLOT_KINDS:
        raise ValueError(f"plot_kind must be one of {', '.join(PLOT_KINDS)}.")

    _configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_summary_csv(summary_csv)
    points, omitted = prepare_plot_points(rows)
    if not points:
        raise ValueError("No finite matched rows available for plotting.")

    x_values, y_values, sessions, amplitudes = _build_swarm_coordinates(points)
    session_colors = _session_colors(plt, sessions)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if plot_kind == "box":
        ax.boxplot(
            _amplitude_group_values(points, amplitudes),
            positions=amplitudes,
            widths=_distribution_width(amplitudes),
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#d9d9d9", "edgecolor": "#555555", "alpha": 0.62},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )
    elif plot_kind == "violin":
        violin = ax.violinplot(
            _violin_group_values(points, amplitudes),
            positions=amplitudes,
            widths=_distribution_width(amplitudes),
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("#d9d9d9")
            body.set_edgecolor("#555555")
            body.set_alpha(0.58)
        if "cmedians" in violin:
            violin["cmedians"].set_color("#111111")
            violin["cmedians"].set_linewidth(1.5)

    _plot_session_points(ax, x_values, y_values, sessions, session_colors)
    _plot_mean_markers(ax, points)
    _style_amplitude_motion_axis(
        ax,
        amplitudes=amplitudes,
        title_label=animal_id,
        plot_label=plot_kind,
        plotted=len(points),
        omitted=omitted,
    )
    ax.legend(title="Session", frameon=False, loc="best")

    resolved_png_path = Path(png_path)
    resolved_svg_path = Path(svg_path)
    resolved_png_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_png_path, dpi=200)
    fig.savefig(resolved_svg_path)
    plt.close(fig)
    return resolved_png_path, resolved_svg_path, len(points), omitted


def _distribution_width(amplitudes: list[float]) -> float:
    if len(amplitudes) > 1:
        min_gap = min(
            right - left for left, right in zip(amplitudes[:-1], amplitudes[1:])
        )
        return max(0.5, min_gap * 0.58)
    return 1.0


def plot_swarm_from_summary(
    summary_csv: str | Path,
    *,
    png_path: str | Path,
    svg_path: str | Path,
    animal_id: str,
) -> tuple[Path, Path, int, int]:
    """Create PNG and SVG swarm plots from a combined summary CSV."""

    return plot_amplitude_motion_from_summary(
        summary_csv,
        png_path=png_path,
        svg_path=svg_path,
        animal_id=animal_id,
        plot_kind="swarm",
    )


def plot_box_from_summary(
    summary_csv: str | Path,
    *,
    png_path: str | Path,
    svg_path: str | Path,
    animal_id: str,
) -> tuple[Path, Path, int, int]:
    """Create PNG and SVG box plots from a summary CSV."""

    return plot_amplitude_motion_from_summary(
        summary_csv,
        png_path=png_path,
        svg_path=svg_path,
        animal_id=animal_id,
        plot_kind="box",
    )


def plot_violin_from_summary(
    summary_csv: str | Path,
    *,
    png_path: str | Path,
    svg_path: str | Path,
    animal_id: str,
) -> tuple[Path, Path, int, int]:
    """Create PNG and SVG violin plots from a summary CSV."""

    return plot_amplitude_motion_from_summary(
        summary_csv,
        png_path=png_path,
        svg_path=svg_path,
        animal_id=animal_id,
        plot_kind="violin",
    )


def _plot_function_for_kind(plot_kind: str):
    if plot_kind == "swarm":
        return plot_swarm_from_summary
    if plot_kind == "box":
        return plot_box_from_summary
    if plot_kind == "violin":
        return plot_violin_from_summary
    raise ValueError(f"Unknown plot kind: {plot_kind}")


def plot_session_distribution_plots(
    output_root: str | Path,
    *,
    plot_kind: str,
    plot_root: str | Path | None = None,
) -> list[tuple[Path, Path, int, int]]:
    """Create one amplitude-motion distribution plot per session summary directory."""

    root = Path(output_root)
    resolved_plot_root = Path(plot_root) if plot_root is not None else default_plot_root(root)
    results: list[tuple[Path, Path, int, int]] = []
    plot_function = _plot_function_for_kind(plot_kind)
    for summary_csv in iter_session_summary_csvs(root):
        session_id = summary_csv.parent.name
        session_plot_root = resolved_plot_root / session_id
        png_path = session_plot_root / f"{session_id}_amplitude_motion_{plot_kind}.png"
        svg_path = session_plot_root / f"{session_id}_amplitude_motion_{plot_kind}.svg"
        results.append(
            plot_function(
                summary_csv,
                png_path=png_path,
                svg_path=svg_path,
                animal_id=session_id,
            )
        )
    return results


def plot_session_swarm_plots(
    output_root: str | Path,
    *,
    plot_root: str | Path | None = None,
) -> list[tuple[Path, Path, int, int]]:
    """Create one swarm plot per session summary directory."""

    return plot_session_distribution_plots(
        output_root,
        plot_kind="swarm",
        plot_root=plot_root,
    )


def main() -> int:
    args = parse_args()
    animal_id = args.animal_root.name
    output_root = args.output_root or Path("analysis") / "stim_motion" / animal_id
    summary_csv = args.summary_csv or output_root / f"{animal_id}_stim_motion_summary.csv"

    if args.rebuild_summary or not summary_csv.exists():
        rows, written_paths = summarize_animal(args.animal_root, output_root=output_root)
        if summary_csv not in written_paths and not summary_csv.exists():
            write_summary_csv(rows, summary_csv)

    plot_root = default_plot_root(output_root)
    print(summary_csv)
    for plot_kind in PLOT_KINDS:
        plot_function = _plot_function_for_kind(plot_kind)
        png_path = (
            args.png
            if plot_kind == "swarm" and args.png
            else plot_root / f"{animal_id}_amplitude_motion_{plot_kind}.png"
        )
        svg_path = (
            args.svg
            if plot_kind == "swarm" and args.svg
            else plot_root / f"{animal_id}_amplitude_motion_{plot_kind}.svg"
        )
        png_path, svg_path, plotted, omitted = plot_function(
            summary_csv,
            png_path=png_path,
            svg_path=svg_path,
            animal_id=animal_id,
        )
        print(
            f"{plot_kind.title()} plot: {plotted} matched stimulations; "
            f"omitted {omitted} unmatched/NaN rows."
        )
        print(png_path)
        print(svg_path)
        if args.per_session:
            session_results = plot_session_distribution_plots(
                output_root,
                plot_kind=plot_kind,
                plot_root=plot_root,
            )
            for session_png, session_svg, session_plotted, session_omitted in session_results:
                print(
                    f"Session {plot_kind}: {session_png.parent.name} "
                    f"({session_plotted} plotted; {session_omitted} omitted)"
                )
                print(session_png)
                print(session_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
