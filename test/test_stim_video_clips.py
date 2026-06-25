import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.stim_video_clips import (
    CameraFrame,
    FrameWindow,
    amplitude_folder_name,
    build_ffmpeg_command,
    clip_output_path,
    discover_camera_inputs,
    frame_window_for_stim,
    load_camera_timeline,
    metadata_fps,
    prepare_session_clip_jobs,
    run_clip_jobs,
)


def _write_metadata(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter=";",
            fieldnames=[
                "frame_camera_serial",
                "frame_camera_name",
                "frame_id",
                "frame_timestamp",
                "frame_image_uid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _camera_row(camera_name, frame_id, timestamp_ns):
    return {
        "frame_camera_serial": "123",
        "frame_camera_name": camera_name,
        "frame_id": str(frame_id),
        "frame_timestamp": str(timestamp_ns),
        "frame_image_uid": str(1000 + frame_id),
    }


class StimVideoClipTests(unittest.TestCase):
    def test_discover_camera_inputs_normal_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "MTEST_2026_01_01_12_00"
            camera_dir = session / f"{session.name}_cameras"
            video_path = camera_dir / "Camera_4.avi"
            camera_dir.mkdir(parents=True)
            video_path.write_bytes(b"video")
            metadata_path = camera_dir / "metadata.csv"
            _write_metadata(metadata_path, [_camera_row("Camera_4", 0, 1_000)])

            inputs = discover_camera_inputs(session, camera_name="Camera_4")

        self.assertEqual(inputs.video_path, video_path)
        self.assertEqual(inputs.metadata_path, metadata_path)

    def test_discover_camera_inputs_irregular_layout_with_root_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "MTEST_2026_01_01_12_00"
            camera_dir = session / "MTEST_2025_01_01_12_00_cameras"
            video_path = camera_dir / "Camera_4.avi"
            camera_dir.mkdir(parents=True)
            video_path.write_bytes(b"video")
            metadata_path = session / "metadata.csv"
            _write_metadata(metadata_path, [_camera_row("Camera_4", 0, 1_000)])

            inputs = discover_camera_inputs(session, camera_name="Camera_4")

        self.assertEqual(inputs.video_path, video_path)
        self.assertEqual(inputs.metadata_path, metadata_path)

    def test_load_camera_timeline_filters_camera_and_uses_video_row_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.csv"
            _write_metadata(
                metadata_path,
                [
                    _camera_row("Camera_3", 0, 9_000_000),
                    _camera_row("Camera_4", 0, 10_000_000),
                    _camera_row("Camera_4", 2, 20_000_000),
                    _camera_row("Camera_3", 1, 21_000_000),
                    _camera_row("Camera_4", 4, 30_000_000),
                ],
            )

            frames = load_camera_timeline(metadata_path, camera_name="Camera_4")

        self.assertEqual([frame.video_index for frame in frames], [0, 1, 2])
        self.assertEqual([frame.frame_id for frame in frames], [0, 2, 4])
        self.assertEqual([frame.relative_ms for frame in frames], [0.0, 10.0, 20.0])

    def test_metadata_fps_uses_median_timestamp_delta(self):
        frames = [
            CameraFrame(0, 0, 0, 0.0),
            CameraFrame(1, 1, 10_000_000, 10.0),
            CameraFrame(2, 2, 20_000_000, 20.0),
            CameraFrame(3, 3, 30_000_000, 30.0),
        ]

        self.assertEqual(metadata_fps(frames), 100.0)

    def test_frame_window_for_stim_maps_pycontrol_ms_to_frame_indices(self):
        frames = [
            CameraFrame(index, index, index * 10_000_000, index * 10.0)
            for index in range(10)
        ]

        window = frame_window_for_stim(
            frames,
            stim_start_ms=30.0,
            stim_end_ms=60.0,
            pre_s=0.02,
            post_s=0.01,
        )

        self.assertEqual(window.start_index, 1)
        self.assertEqual(window.end_index_exclusive, 7)
        self.assertEqual(window.clip_start_ms, 10.0)
        self.assertEqual(window.clip_end_ms, 70.0)
        self.assertEqual(window.stim_on_output_s, 0.02)
        self.assertEqual(window.stim_off_output_s, 0.05)

    def test_amplitude_folder_name(self):
        self.assertEqual(amplitude_folder_name("0", "True"), "000_uA_sham")
        self.assertEqual(amplitude_folder_name("25.0", "False"), "025_uA")
        self.assertEqual(amplitude_folder_name("12.5", ""), "12p5_uA")

    def test_clip_output_path_is_inside_session_stim_videos_amplitude_folder(self):
        row = {
            "event_number": "2",
            "amplitude_uA": "25.0",
            "is_sham": "False",
            "stim_event_index_onset": "9",
        }
        session = Path("/tmp/MTEST_2026_01_01_12_00")

        path = clip_output_path(session, row, pre_s=2.0, post_s=2.0)

        self.assertEqual(
            path,
            session
            / "stim_videos"
            / "025_uA"
            / "event_002_025_uA_onset_9_pre2s_post2s.mp4",
        )

    def test_build_ffmpeg_command_contains_frame_trim_and_annotations(self):
        row = {
            "event_number": "2",
            "command": "25_uA",
            "amplitude_uA": "25.0",
            "is_sham": "False",
        }
        window = FrameWindow(
            start_index=10,
            end_index_exclusive=50,
            clip_start_ms=100.0,
            clip_end_ms=500.0,
            stim_on_output_s=0.2,
            stim_off_output_s=0.35,
            fps=100.0,
        )

        command = build_ffmpeg_command(
            "/tmp/input.avi",
            "/tmp/output.mp4",
            row,
            window,
            session_id="MTEST",
        )

        vf = command[command.index("-vf") + 1]
        self.assertIn("between(n\\,10\\,49)", vf)
        self.assertIn("setpts=N/(100*TB)", vf)
        self.assertIn("MTEST", vf)
        self.assertIn("25_uA", vf)
        self.assertIn("STIM ON", vf)
        self.assertEqual(command[-1], "/tmp/output.mp4")

    def test_prepare_session_clip_jobs_and_dry_run_do_not_create_output_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "MTEST_2026_01_01_12_00"
            camera_dir = session / f"{session.name}_cameras"
            video_path = camera_dir / "Camera_4.avi"
            camera_dir.mkdir(parents=True)
            video_path.write_bytes(b"video")
            _write_metadata(
                camera_dir / "metadata.csv",
                [_camera_row("Camera_4", index, index * 10_000_000) for index in range(20)],
            )
            summary_csv = Path(temp_dir) / "summary.csv"
            with summary_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "session_id",
                        "event_number",
                        "command",
                        "is_sham",
                        "amplitude_uA",
                        "stim_event_index_onset",
                        "window_start_ms",
                        "window_end_ms",
                        "matched_pycontrol_window",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "session_id": session.name,
                        "event_number": "1",
                        "command": "25_uA",
                        "is_sham": "False",
                        "amplitude_uA": "25.0",
                        "stim_event_index_onset": "6",
                        "window_start_ms": "40.0",
                        "window_end_ms": "70.0",
                        "matched_pycontrol_window": "True",
                    }
                )

            jobs = prepare_session_clip_jobs(
                session,
                summary_csv,
                camera_name="Camera_4",
                pre_s=0.01,
                post_s=0.01,
            )
            with mock.patch("builtins.print"):
                result = run_clip_jobs(jobs, dry_run=True)

            self.assertEqual(len(jobs), 1)
            self.assertEqual(result.prepared, 1)
            self.assertEqual(result.written, 0)
            self.assertFalse((session / "stim_videos").exists())


if __name__ == "__main__":
    unittest.main()
