import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.stim_motion.plot_stim_motion_swarm import (
    PlotPoint,
    compute_amplitude_means,
    compute_swarm_offsets,
    plot_box_from_summary,
    plot_session_swarm_plots,
    plot_swarm_from_summary,
    plot_violin_from_summary,
    prepare_plot_points,
)
from scripts.stim_motion.plot_stim_motion_trajectories import (
    build_motion_traces,
    extract_motion_trace,
    plot_trajectories_from_summary,
    traces_to_matrix,
    trajectory_grid,
)
from utils.stim_event_log import FIELDNAMES as STIM_FIELDNAMES
from utils.stim_motion_summary import (
    CM_PER_MOTION_COUNT,
    SUMMARY_FIELDNAMES,
    ExpectedMarker,
    StimAttempt,
    align_pycontrol_markers,
    amplitude_mA_to_uA,
    load_session_motion_data,
    load_analog_pca,
    motion_counts_to_cm,
    parse_pycontrol_markers,
    session_summary_csv_name,
    summarize_motion_window,
    summarize_session,
)
from utils.stim_motion_summary import PyControlMarker


def _stim_row(**updates):
    row = {field: "" for field in STIM_FIELDNAMES}
    row.update(
        {
            "session_id": "MTEST_2026_01_01_12_00",
            "source": "timed_random",
            "experiment": "m2_random_timed",
            "profile": "m2_timing",
            "pulse_mode": "train",
            "channels": "[1, 2]",
            "channel_labels": '["left", "right"]',
            "physical_contacts": '["unknown", "unknown"]',
            "freq_hz": "150",
            "pw_ms": "[0.2, 0.2]",
            "duration_s": "3.0",
            "inter_phase_s": "5e-05",
            "stim_port": "COM8",
            "mock_mode": "False",
        }
    )
    row.update(updates)
    return row


def _write_stim_events(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STIM_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDNAMES})


class StimMotionSummaryTests(unittest.TestCase):
    def test_summary_fieldnames_exclude_redundant_stim_metadata(self):
        removed_fields = {
            "animal_id",
            "event",
            "experiment",
            "profile",
            "experiment_profile",
            "pulse_mode",
            "channels",
            "channel_labels",
            "physical_contacts",
            "freq_hz",
            "pw_ms",
            "amp_mA",
            "inter_phase_s",
            "condition_index",
            "pycontrol_onset_ms",
            "pycontrol_offset_ms",
            "scheduled_onset_s",
            "actual_onset_s",
            "stim_wall_time_iso_onset",
            "stim_wall_time_iso_offset",
            "match_status",
            "motion_x_signed_sum",
            "motion_y_signed_sum",
            "motion_magnitude_sum",
            "motion_x_abs_sum",
            "motion_y_abs_sum",
            "motion_x_abs_cm",
            "motion_y_abs_cm",
        }

        self.assertTrue(removed_fields.isdisjoint(SUMMARY_FIELDNAMES))

    def test_amplitude_mA_to_uA_handles_scalar_and_multichannel_cells(self):
        self.assertEqual(amplitude_mA_to_uA("0.025"), 25.0)
        self.assertEqual(amplitude_mA_to_uA("[0.05, 0.06, 0.05]"), 60.0)
        self.assertEqual(amplitude_mA_to_uA("[0, 0]"), 0.0)
        self.assertTrue(math.isnan(amplitude_mA_to_uA("")))

    def test_load_analog_pca_and_motion_distance_cm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pca_path = Path(temp_dir) / "motion.pca"
            expected = np.asarray([[0, 3], [10, 4], [20, -5]], dtype="<i4")
            expected.tofile(pca_path)

            loaded = load_analog_pca(pca_path)
            np.testing.assert_array_equal(loaded, expected)

        x_data = np.asarray([[0, 3], [10, 0], [20, 5]], dtype="<i4")
        y_data = np.asarray([[0, 4], [10, -2], [20, 12]], dtype="<i4")
        summary = summarize_motion_window(x_data, y_data, 0, 20)

        self.assertAlmostEqual(CM_PER_MOTION_COUNT, 2.54 / 595.7)
        self.assertEqual(summary["motion_sample_count"], 2)
        self.assertAlmostEqual(summary["motion_distance_cm"], motion_counts_to_cm(7.0))

    def test_parse_pycontrol_markers_treats_code_104_unknown_as_sham_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "session.txt"
            log_path.write_text(
                "\n".join(
                    [
                        "I Start date : 2026/01/01 12:00:00",
                        "P 1001 1000, external_stim_marker code=104 name=unknown count=1 active=False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            markers = parse_pycontrol_markers(log_path)

        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].marker_time_ms, 1000)
        self.assertEqual(markers[0].code, 104)
        self.assertEqual(markers[0].name, "unknown")

    def test_align_pycontrol_markers_skips_missing_leading_and_internal_events(self):
        attempts = [
            StimAttempt(
                onset_row={},
                offset_row={},
                details={},
                event_number=1,
                is_sham=False,
                expected_markers=[
                    ExpectedMarker(0, "on", 101, -9000),
                    ExpectedMarker(0, "off", 102, -6000),
                ],
            ),
            StimAttempt(
                onset_row={},
                offset_row={},
                details={},
                event_number=2,
                is_sham=False,
                expected_markers=[
                    ExpectedMarker(1, "on", 101, 2000),
                    ExpectedMarker(1, "off", 102, 5000),
                ],
            ),
            StimAttempt(
                onset_row={},
                offset_row=None,
                details={},
                event_number=3,
                is_sham=True,
                expected_markers=[ExpectedMarker(2, "sham", 104, 17000)],
            ),
            StimAttempt(
                onset_row={},
                offset_row=None,
                details={},
                event_number=4,
                is_sham=True,
                expected_markers=[ExpectedMarker(3, "sham", 104, 32000)],
            ),
            StimAttempt(
                onset_row={},
                offset_row={},
                details={},
                event_number=5,
                is_sham=False,
                expected_markers=[
                    ExpectedMarker(4, "on", 101, 47000),
                    ExpectedMarker(4, "off", 102, 50000),
                ],
            ),
        ]
        markers = [
            PyControlMarker(2601, 2600, 101, "stim_on", 1, "True"),
            PyControlMarker(5601, 5600, 102, "stim_off", 2, "False"),
            PyControlMarker(17601, 17600, 104, "unknown", 3, "False"),
            PyControlMarker(47601, 47600, 101, "stim_on", 4, "True"),
            PyControlMarker(50601, 50600, 102, "stim_off", 5, "False"),
        ]

        aligned = align_pycontrol_markers(attempts, markers)

        self.assertNotIn((0, "on"), aligned)
        self.assertEqual(aligned[(1, "on")].marker_time_ms, 2600)
        self.assertEqual(aligned[(1, "off")].marker_time_ms, 5600)
        self.assertEqual(aligned[(2, "sham")].marker_time_ms, 17600)
        self.assertNotIn((3, "sham"), aligned)
        self.assertEqual(aligned[(4, "on")].marker_time_ms, 47600)
        self.assertEqual(aligned[(4, "off")].marker_time_ms, 50600)

    def test_summarize_session_filters_to_pycontrol_window_and_sums_motion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "MTEST_2026_01_01_12_00"
            session_dir.mkdir()
            stim_path = session_dir / "stim_events.csv"
            _write_stim_events(
                stim_path,
                [
                    _stim_row(
                        event_index="1",
                        event="stim_on",
                        command="25_uA",
                        pycontrol_code="101",
                        amp_mA="[0.025, 0.025]",
                        wall_time_iso="2026-01-01T11:59:50",
                        details_json='{"event_number": 1, "condition_index": 2}',
                    ),
                    _stim_row(
                        event_index="2",
                        event="stim_off",
                        command="25_uA",
                        pycontrol_code="102",
                        amp_mA="[0, 0]",
                        wall_time_iso="2026-01-01T11:59:53",
                        details_json='{"event_number": 1, "condition_index": 2}',
                    ),
                    _stim_row(
                        event_index="3",
                        event="stim_on",
                        command="50_uA",
                        pycontrol_code="101",
                        amp_mA="[0.05, 0.05]",
                        wall_time_iso="2026-01-01T12:00:02",
                        details_json='{"event_number": 2, "condition_index": 3}',
                    ),
                    _stim_row(
                        event_index="4",
                        event="stim_off",
                        command="50_uA",
                        pycontrol_code="102",
                        amp_mA="[0, 0]",
                        wall_time_iso="2026-01-01T12:00:05",
                        details_json='{"event_number": 2, "condition_index": 3}',
                    ),
                    _stim_row(
                        event_index="5",
                        event="stim_sham",
                        command="0_uA",
                        pycontrol_code="104",
                        amp_mA="[0, 0]",
                        wall_time_iso="2026-01-01T12:00:10",
                        details_json='{"event_number": 3, "condition_index": 1}',
                    ),
                    _stim_row(
                        event_index="6",
                        event="stim_sham",
                        command="0_uA",
                        pycontrol_code="104",
                        amp_mA="[0, 0]",
                        wall_time_iso="2026-01-01T12:00:20",
                        details_json='{"event_number": 4, "condition_index": 1}',
                    ),
                ],
            )
            log_path = session_dir / "MTEST_2026_01_01_12_00-2026-01-01-120000.txt"
            log_path.write_text(
                "\n".join(
                    [
                        "I Start date : 2026/01/01 12:00:00",
                        'S {"trial": 2}',
                        'E {"motion": 6, "cursor_update": 4}',
                        "P 2501 2500, external_stim_marker code=101 name=stim_on count=1 active=True",
                        "P 5501 5500, external_stim_marker code=102 name=stim_off count=2 active=False",
                        "P 10501 10500, external_stim_marker code=104 name=unknown count=3 active=False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            x_data = np.asarray(
                [[2500, 3], [3500, 0], [5500, 99], [10500, 5], [13000, 0]],
                dtype="<i4",
            )
            y_data = np.asarray(
                [[2500, 4], [3500, -1], [5500, 99], [10500, 12], [13000, -2]],
                dtype="<i4",
            )
            x_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-X.pca")
            y_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-Y.pca")

            rows = summarize_session(session_dir, animal_id="MTEST")

        self.assertEqual([row["event_number"] for row in rows], [2, 3])
        self.assertEqual(rows[0]["amplitude_uA"], 50.0)
        self.assertEqual(rows[0]["matched_pycontrol_window"], "True")
        self.assertAlmostEqual(rows[0]["motion_distance_cm"], motion_counts_to_cm(6.0))
        self.assertEqual(rows[1]["matched_pycontrol_window"], "True")
        self.assertEqual(rows[1]["window_start_ms"], 10500.0)
        self.assertEqual(rows[1]["window_end_ms"], 13500.0)
        self.assertAlmostEqual(rows[1]["motion_distance_cm"], motion_counts_to_cm(15.0))

    def test_load_session_motion_data_finds_matching_pca_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "MTEST_2026_01_01_12_00"
            session_dir.mkdir()
            log_path = session_dir / "MTEST_2026_01_01_12_00-2026-01-01-120000.txt"
            log_path.write_text("I Start date : 2026/01/01 12:00:00\n", encoding="utf-8")
            x_data = np.asarray([[0, 1], [10, 2]], dtype="<i4")
            y_data = np.asarray([[0, 3], [10, 4]], dtype="<i4")
            x_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-X.pca")
            y_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-Y.pca")

            loaded_x, loaded_y = load_session_motion_data(session_dir)

        np.testing.assert_array_equal(loaded_x, x_data)
        np.testing.assert_array_equal(loaded_y, y_data)


class StimMotionPlotTests(unittest.TestCase):
    def test_swarm_offsets_are_deterministic_and_bounded(self):
        offsets = compute_swarm_offsets(5, max_width=0.4)

        self.assertEqual(offsets, [0.0, -0.2, 0.2, -0.4, 0.4])
        self.assertLessEqual(max(abs(offset) for offset in offsets), 0.4)
        self.assertEqual(offsets, compute_swarm_offsets(5, max_width=0.4))

    def test_compute_amplitude_means_groups_by_amplitude(self):
        means = compute_amplitude_means(
            [
                PlotPoint(25.0, 10.0, "s1", 1),
                PlotPoint(25.0, 14.0, "s1", 2),
                PlotPoint(50.0, 5.0, "s2", 3),
            ]
        )

        self.assertEqual(means, [(25.0, 12.0), (50.0, 5.0)])

    def test_prepare_plot_points_omits_unmatched_and_nan_motion_rows(self):
        rows = [
            {
                "matched_pycontrol_window": "True",
                "amplitude_uA": "25",
                "motion_distance_cm": "10",
                "session_id": "s1",
                "event_number": "1",
            },
            {
                "matched_pycontrol_window": "False",
                "amplitude_uA": "25",
                "motion_distance_cm": "12",
                "session_id": "s1",
                "event_number": "2",
            },
            {
                "matched_pycontrol_window": "True",
                "amplitude_uA": "50",
                "motion_distance_cm": "",
                "session_id": "s1",
                "event_number": "3",
            },
        ]

        points, omitted = prepare_plot_points(rows)

        self.assertEqual(len(points), 1)
        self.assertEqual(omitted, 2)
        self.assertEqual(points[0].amplitude_uA, 25.0)

    def test_extract_motion_trace_and_interpolate_matrix(self):
        x_data = np.asarray([[100, 3], [110, 0], [120, 5]], dtype="<i4")
        y_data = np.asarray([[100, 4], [110, -2], [120, 12]], dtype="<i4")

        trace = extract_motion_trace(
            x_data,
            y_data,
            start_ms=100,
            end_ms=125,
            amplitude_uA=50,
            session_id="s1",
            event_number=1,
        )

        self.assertIsNotNone(trace)
        np.testing.assert_array_equal(trace.relative_time_ms, np.asarray([0, 10, 20]))
        np.testing.assert_allclose(
            trace.motion_magnitude_cm,
            motion_counts_to_cm(np.asarray([5, 2, 13], dtype=float)),
        )
        grid = trajectory_grid([trace], time_step_ms=10)
        matrix = traces_to_matrix([trace], grid)
        np.testing.assert_array_equal(grid, np.asarray([0, 10, 20]))
        np.testing.assert_allclose(
            matrix,
            motion_counts_to_cm(np.asarray([[5, 2, 13]], dtype=float)),
        )

    def test_build_motion_traces_omits_unmatched_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "s1"
            session_dir.mkdir()
            log_path = session_dir / "s1-2026-01-01-120000.txt"
            log_path.write_text("I Start date : 2026/01/01 12:00:00\n", encoding="utf-8")
            x_data = np.asarray([[100, 3], [110, 0]], dtype="<i4")
            y_data = np.asarray([[100, 4], [110, -2]], dtype="<i4")
            x_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-X.pca")
            y_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-Y.pca")

            traces, omitted = build_motion_traces(
                [
                    {
                        "session_id": "s1",
                        "event_number": "1",
                        "matched_pycontrol_window": "True",
                        "amplitude_uA": "50",
                        "window_start_ms": "100",
                        "window_end_ms": "120",
                    },
                    {
                        "session_id": "s1",
                        "event_number": "2",
                        "matched_pycontrol_window": "False",
                        "amplitude_uA": "50",
                        "window_start_ms": "",
                        "window_end_ms": "",
                    },
                ],
                animal_root=root,
            )

        self.assertEqual(len(traces), 1)
        self.assertEqual(omitted, 1)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_distribution_plots_from_summary_smoke_write_png_and_svg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_csv = root / "summary.csv"
            _write_summary(
                summary_csv,
                [
                    {
                        "animal_id": "MTEST",
                        "session_id": "s1",
                        "event_number": "1",
                        "matched_pycontrol_window": "True",
                        "amplitude_uA": "0",
                        "motion_distance_cm": "5",
                    },
                    {
                        "animal_id": "MTEST",
                        "session_id": "s2",
                        "event_number": "2",
                        "matched_pycontrol_window": "True",
                        "amplitude_uA": "25",
                        "motion_distance_cm": "8",
                    },
                    {
                        "animal_id": "MTEST",
                        "session_id": "s2",
                        "event_number": "3",
                        "matched_pycontrol_window": "False",
                        "amplitude_uA": "25",
                        "motion_distance_cm": "",
                    },
                ],
            )

            plot_functions = [
                ("swarm", plot_swarm_from_summary),
                ("box", plot_box_from_summary),
                ("violin", plot_violin_from_summary),
            ]
            for plot_name, plot_function in plot_functions:
                with self.subTest(plot_name=plot_name):
                    png_path, svg_path, plotted, omitted = plot_function(
                        summary_csv,
                        png_path=root / f"{plot_name}.png",
                        svg_path=root / f"{plot_name}.svg",
                        animal_id="MTEST",
                    )

                    self.assertEqual(plotted, 2)
                    self.assertEqual(omitted, 1)
                    self.assertTrue(png_path.exists())
                    self.assertTrue(svg_path.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plot_trajectories_from_summary_smoke_writes_png_and_svg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "s1"
            session_dir.mkdir()
            log_path = session_dir / "s1-2026-01-01-120000.txt"
            log_path.write_text("I Start date : 2026/01/01 12:00:00\n", encoding="utf-8")
            x_data = np.asarray([[100, 3], [110, 0], [120, 5]], dtype="<i4")
            y_data = np.asarray([[100, 4], [110, -2], [120, 12]], dtype="<i4")
            x_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-X.pca")
            y_data.tofile(session_dir / f"{log_path.with_suffix('').name}_MotSen1-Y.pca")
            summary_csv = root / "summary.csv"
            _write_summary(
                summary_csv,
                [
                    {
                        "animal_id": "MTEST",
                        "session_id": "s1",
                        "event_number": "1",
                        "matched_pycontrol_window": "True",
                        "amplitude_uA": "50",
                        "window_start_ms": "100",
                        "window_end_ms": "125",
                    }
                ],
            )

            png_path, svg_path, plotted, omitted = plot_trajectories_from_summary(
                summary_csv,
                animal_root=root,
                png_path=root / "trajectory.png",
                svg_path=root / "trajectory.svg",
                title_label="MTEST",
            )

            self.assertEqual(plotted, 1)
            self.assertEqual(omitted, 0)
            self.assertTrue(png_path.exists())
            self.assertTrue(svg_path.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plot_session_swarm_plots_writes_one_plot_per_session_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "MTEST_2026_01_01_12_00"
            session_dir.mkdir()
            _write_summary(
                session_dir / f"{session_dir.name}_stim_motion_summary.csv",
                [
                    {
                        "animal_id": "MTEST",
                        "session_id": "MTEST_2026_01_01_12_00",
                        "event_number": "1",
                        "matched_pycontrol_window": "True",
                        "amplitude_uA": "25",
                        "motion_distance_cm": "7",
                    }
                ],
            )

            results = plot_session_swarm_plots(root)

            self.assertEqual(len(results), 1)
            png_path, svg_path, plotted, omitted = results[0]
            self.assertEqual(plotted, 1)
            self.assertEqual(omitted, 0)
            self.assertTrue(png_path.exists())
            self.assertTrue(svg_path.exists())
            self.assertEqual(png_path.parent, root / "plots" / session_dir.name)
            self.assertEqual(svg_path.parent, root / "plots" / session_dir.name)

    def test_session_summary_csv_name_includes_session_id(self):
        self.assertEqual(
            session_summary_csv_name("MTEST_2026_01_01_12_00"),
            "MTEST_2026_01_01_12_00_stim_motion_summary.csv",
        )


if __name__ == "__main__":
    unittest.main()
