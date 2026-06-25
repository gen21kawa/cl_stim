"""Plot stimulation-aligned MotSen1 motion trajectories."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stim_motion_summary import (
    load_session_motion_data,
    motion_counts_to_cm,
    summarize_animal,
    write_summary_csv,
)
from scripts.stim_motion.plot_stim_motion_swarm import iter_session_summary_csvs


DEFAULT_ANIMAL_ROOT = PROJECT_ROOT / "data" / "raw" / "M114"
DEFAULT_TIME_STEP_MS = 10.0


@dataclass(frozen=True)
class MotionTrace:
    """One stimulation-aligned MotSen1 trace."""

    amplitude_uA: float
    session_id: str
    event_number: int
    relative_time_ms: np.ndarray
    motion_magnitude_cm: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean +/- std MotSen1 trajectories aligned to stimulation onset."
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
        "--time-step-ms",
        type=float,
        default=DEFAULT_TIME_STEP_MS,
        help="Common trajectory time grid step in milliseconds.",
    )
    parser.add_argument(
        "--per-session",
        action="store_true",
        help="Also write one trajectory plot per session summary CSV.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <output-root>/plots/<animal>_motion_trajectories.png.",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="SVG output path. Defaults to <output-root>/plots/<animal>_motion_trajectories.svg.",
    )
    return parser.parse_args()


def default_plot_root(output_root: str | Path) -> Path:
    """Return the default folder for generated figures."""

    return Path(output_root) / "plots"


def read_summary_csv(summary_csv: str | Path) -> list[dict[str, str]]:
    with Path(summary_csv).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _paired_window_values(
    x_data: np.ndarray,
    y_data: np.ndarray,
    start_ms: float,
    end_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_data.size == 0 or y_data.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    if len(x_data) == len(y_data) and np.array_equal(x_data[:, 0], y_data[:, 0]):
        mask = (x_data[:, 0] >= start_ms) & (x_data[:, 0] < end_ms)
        times = x_data[mask, 0].astype(float)
        x_values = x_data[mask, 1].astype(float)
        y_values = y_data[mask, 1].astype(float)
        return times, x_values, y_values

    y_by_time = {int(time_ms): float(value) for time_ms, value in y_data}
    x_window = x_data[(x_data[:, 0] >= start_ms) & (x_data[:, 0] < end_ms)]
    paired = [
        (float(time_ms), float(x_value), y_by_time[int(time_ms)])
        for time_ms, x_value in x_window
        if int(time_ms) in y_by_time
    ]
    if not paired:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    times, x_values, y_values = (np.asarray(values, dtype=float) for values in zip(*paired))
    return times, x_values, y_values


def extract_motion_trace(
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    start_ms: float,
    end_ms: float,
    amplitude_uA: float,
    session_id: str,
    event_number: int,
) -> MotionTrace | None:
    """Extract one stimulation-aligned motion magnitude trace."""

    if not (math.isfinite(start_ms) and math.isfinite(end_ms)) or end_ms <= start_ms:
        return None
    times, x_values, y_values = _paired_window_values(x_data, y_data, start_ms, end_ms)
    if times.size == 0:
        return None
    magnitude_cm = motion_counts_to_cm(np.sqrt((x_values * x_values) + (y_values * y_values)))
    return MotionTrace(
        amplitude_uA=amplitude_uA,
        session_id=session_id,
        event_number=event_number,
        relative_time_ms=times - start_ms,
        motion_magnitude_cm=magnitude_cm,
    )


def build_motion_traces(
    summary_rows: list[dict[str, object]],
    *,
    animal_root: str | Path,
) -> tuple[list[MotionTrace], int]:
    """Build stimulation-window traces from summary rows and raw motion files."""

    root = Path(animal_root)
    motion_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    traces: list[MotionTrace] = []
    omitted = 0

    for row in summary_rows:
        if str(row.get("matched_pycontrol_window", "")).lower() != "true":
            omitted += 1
            continue
        amplitude = _as_finite_float(row.get("amplitude_uA"))
        start_ms = _as_finite_float(row.get("window_start_ms"))
        end_ms = _as_finite_float(row.get("window_end_ms"))
        session_id = str(row.get("session_id", ""))
        if amplitude is None or start_ms is None or end_ms is None or not session_id:
            omitted += 1
            continue
        if session_id not in motion_cache:
            motion_cache[session_id] = load_session_motion_data(root / session_id)
        trace = extract_motion_trace(
            *motion_cache[session_id],
            start_ms=start_ms,
            end_ms=end_ms,
            amplitude_uA=amplitude,
            session_id=session_id,
            event_number=_as_int(row.get("event_number")),
        )
        if trace is None:
            omitted += 1
            continue
        traces.append(trace)

    return traces, omitted


def trajectory_grid(traces: list[MotionTrace], *, time_step_ms: float) -> np.ndarray:
    """Return a common time grid covering all traces."""

    if not traces:
        return np.asarray([], dtype=float)
    if not math.isfinite(time_step_ms) or time_step_ms <= 0:
        raise ValueError("--time-step-ms must be finite and greater than zero.")
    max_time_ms = max(float(trace.relative_time_ms[-1]) for trace in traces)
    return np.arange(0.0, max_time_ms + (time_step_ms * 0.5), time_step_ms)


def traces_to_matrix(
    traces: list[MotionTrace],
    grid_ms: np.ndarray,
) -> np.ndarray:
    """Interpolate traces onto ``grid_ms`` and leave out-of-window samples NaN."""

    matrix = np.full((len(traces), len(grid_ms)), np.nan, dtype=float)
    for index, trace in enumerate(traces):
        if trace.relative_time_ms.size == 0:
            continue
        start = float(trace.relative_time_ms[0])
        end = float(trace.relative_time_ms[-1])
        valid = (grid_ms >= start) & (grid_ms <= end)
        if not np.any(valid):
            continue
        matrix[index, valid] = np.interp(
            grid_ms[valid],
            trace.relative_time_ms,
            trace.motion_magnitude_cm,
        )
    return matrix


def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "cl_stim_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def plot_trajectories_from_summary(
    summary_csv: str | Path,
    *,
    animal_root: str | Path,
    png_path: str | Path,
    svg_path: str | Path,
    title_label: str,
    time_step_ms: float = DEFAULT_TIME_STEP_MS,
) -> tuple[Path, Path, int, int]:
    """Create mean +/- std trajectory plots from a summary CSV."""

    _configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_summary_csv(summary_csv)
    traces, omitted = build_motion_traces(rows, animal_root=animal_root)
    if not traces:
        raise ValueError("No finite matched trajectory rows available for plotting.")

    grid_ms = trajectory_grid(traces, time_step_ms=time_step_ms)
    amplitudes = sorted({trace.amplitude_uA for trace in traces})
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for index, amplitude in enumerate(amplitudes):
        group = [trace for trace in traces if trace.amplitude_uA == amplitude]
        matrix = traces_to_matrix(group, grid_ms)
        valid = np.any(np.isfinite(matrix), axis=0)
        if not np.any(valid):
            continue
        valid_matrix = matrix[:, valid]
        mean = np.nanmean(valid_matrix, axis=0)
        std = np.nanstd(valid_matrix, axis=0)
        time_s = grid_ms[valid] / 1000.0
        color = color_cycle[index % len(color_cycle)]
        label = f"{amplitude:g} uA (n={len(group)})"
        ax.plot(time_s, mean, color=color, linewidth=2.0, label=label)
        lower = np.maximum(mean - std, 0.0)
        ax.fill_between(
            time_s,
            lower,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Time from stimulation onset (s)")
    ax.set_ylabel("Motion distance (cm/sample)")
    ax.set_title(
        f"{title_label}: MotSen1 trajectories\n"
        f"Mean +/- std; {len(traces)} matched stimulations plotted; {omitted} omitted"
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Amplitude", frameon=False, loc="best")

    resolved_png_path = Path(png_path)
    resolved_svg_path = Path(svg_path)
    resolved_png_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_png_path, dpi=200)
    fig.savefig(resolved_svg_path)
    plt.close(fig)
    return resolved_png_path, resolved_svg_path, len(traces), omitted


def plot_session_trajectory_plots(
    animal_root: str | Path,
    output_root: str | Path,
    *,
    plot_root: str | Path | None = None,
    time_step_ms: float = DEFAULT_TIME_STEP_MS,
) -> list[tuple[Path, Path, int, int]]:
    """Create one mean +/- std trajectory plot per session summary CSV."""

    root = Path(output_root)
    resolved_plot_root = Path(plot_root) if plot_root is not None else default_plot_root(root)
    results: list[tuple[Path, Path, int, int]] = []
    for summary_csv in iter_session_summary_csvs(root):
        session_id = summary_csv.parent.name
        session_plot_root = resolved_plot_root / session_id
        png_path = session_plot_root / f"{session_id}_motion_trajectories.png"
        svg_path = session_plot_root / f"{session_id}_motion_trajectories.svg"
        results.append(
            plot_trajectories_from_summary(
                summary_csv,
                animal_root=animal_root,
                png_path=png_path,
                svg_path=svg_path,
                title_label=session_id,
                time_step_ms=time_step_ms,
            )
        )
    return results


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
    png_path = args.png or plot_root / f"{animal_id}_motion_trajectories.png"
    svg_path = args.svg or plot_root / f"{animal_id}_motion_trajectories.svg"
    png_path, svg_path, plotted, omitted = plot_trajectories_from_summary(
        summary_csv,
        animal_root=args.animal_root,
        png_path=png_path,
        svg_path=svg_path,
        title_label=animal_id,
        time_step_ms=args.time_step_ms,
    )

    print(f"Plotted {plotted} matched trajectories; omitted {omitted} rows.")
    print(summary_csv)
    print(png_path)
    print(svg_path)
    if args.per_session:
        session_results = plot_session_trajectory_plots(
            args.animal_root,
            output_root,
            plot_root=plot_root,
            time_step_ms=args.time_step_ms,
        )
        for session_png, session_svg, session_plotted, session_omitted in session_results:
            print(
                f"Session trajectories: {session_png.parent.name} "
                f"({session_plotted} plotted; {session_omitted} omitted)"
            )
            print(session_png)
            print(session_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
