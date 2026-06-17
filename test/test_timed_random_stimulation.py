import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import timed_random_stimulation as timed_random


def base_config(protocol):
    return {
        "run": {"mock_mode": True, "active_timed_random_protocol": "protocol"},
        "hardware": {"stimulator_port": "MOCK"},
        "pycontrol_events": {
            "enabled": False,
            "port": "MOCK_PYCONTROL",
            "baud_rate": 9600,
            "codes": {
                "session_start": 110,
                "session_end": 111,
                "stim_on": 101,
                "stim_off": 102,
                "stim_pulse": 103,
                "stim_sham": 104,
            },
        },
        "stimulation": {
            "train_profile": {
                "freq": 50,
                "amp": [0.05, 0.06],
                "pw": 0.2,
                "duration": 0.2,
                "pulse_mode": "train",
                "channel": [1, 2],
                "inter_phase": 50e-6,
            },
            "single_profile": {
                "freq": 50,
                "amp": 0.05,
                "pw": 0.2,
                "duration": 0.2,
                "pulse_mode": "single_pulse",
                "channel": [1],
                "inter_phase": 50e-6,
            },
        },
        "timed_random_protocols": {"protocol": protocol},
    }


def valid_protocol(**updates):
    protocol = {
        "profile": "train_profile",
        "interval_s": 1.0,
        "real_enabled": False,
        "conditions": [
            {"name": "sham", "sham": True, "amp": [0, 0], "duration": 0.1},
            {"name": "stim", "amp": 0.025, "duration": 0.1},
        ],
    }
    protocol.update(updates)
    return protocol


def validate_with(protocol):
    config = base_config(protocol)
    with mock.patch.object(timed_random, "CONFIG", config):
        return timed_random.validate_timed_random_protocol("protocol", protocol)


class TimedRandomValidationTests(unittest.TestCase):
    def test_unknown_profile_fails(self):
        protocol = valid_protocol(profile="missing")
        with self.assertRaisesRegex(ValueError, "unknown stimulation profile"):
            validate_with(protocol)

    def test_non_train_profile_fails(self):
        protocol = valid_protocol(profile="single_profile")
        with self.assertRaisesRegex(ValueError, "requires a train-mode profile"):
            validate_with(protocol)

    def test_missing_or_empty_conditions_fail(self):
        for conditions in ([], None):
            with self.subTest(conditions=conditions):
                protocol = valid_protocol(conditions=conditions)
                with self.assertRaisesRegex(ValueError, "at least one condition"):
                    validate_with(protocol)

    def test_duration_must_be_shorter_than_interval(self):
        protocol = valid_protocol(
            interval_s=0.1,
            conditions=[{"name": "too_long", "amp": 0.01, "duration": 0.1}],
        )
        with self.assertRaisesRegex(ValueError, "must be shorter than interval"):
            validate_with(protocol)

    def test_amp_above_profile_max_fails(self):
        protocol = valid_protocol(
            conditions=[{"name": "too_high", "amp": [0.05, 0.061], "duration": 0.1}]
        )
        with self.assertRaisesRegex(ValueError, "exceeds the profile maximum"):
            validate_with(protocol)

    def test_scalar_per_channel_expansion_and_sham_detection(self):
        protocol = valid_protocol(
            conditions=[
                {"name": "scalar", "amp": 0.025, "duration": 0.1},
                {"name": "per_channel", "amp": [0.01, 0.02], "duration": 0.1},
                {"name": "zero_sham", "amp": [0, 0], "duration": 0.1},
                {
                    "name": "flagged_sham",
                    "sham": True,
                    "amp": [0.01, 0.02],
                    "duration": 0.1,
                },
            ]
        )

        validated = validate_with(protocol)
        conditions = validated["conditions"]

        self.assertEqual(conditions[0].amp_values, [0.025, 0.025])
        self.assertFalse(conditions[0].sham)
        self.assertEqual(conditions[1].amp_values, [0.01, 0.02])
        self.assertFalse(conditions[1].sham)
        self.assertTrue(conditions[2].sham)
        self.assertTrue(conditions[3].sham)


class TimedRandomShuffleTests(unittest.TestCase):
    def test_balanced_shuffle_reproducible_with_seed(self):
        first = timed_random.BalancedConditionDeck(4, random.Random(123))
        second = timed_random.BalancedConditionDeck(4, random.Random(123))

        first_draws = [first.next_index() for _ in range(8)]
        second_draws = [second.next_index() for _ in range(8)]

        self.assertEqual(first_draws, second_draws)
        self.assertEqual(sorted(index for index, cycle in first_draws[:4]), [0, 1, 2, 3])
        self.assertEqual(sorted(index for index, cycle in first_draws[4:]), [0, 1, 2, 3])
        self.assertEqual({cycle for index, cycle in first_draws[:4]}, {1})
        self.assertEqual({cycle for index, cycle in first_draws[4:]}, {2})


class TimedRandomSmokeTests(unittest.TestCase):
    def test_mock_mode_smoke_writes_logs(self):
        protocol = valid_protocol(
            interval_s=0.005,
            conditions=[
                {"name": "sham", "sham": True, "amp": [0, 0], "duration": 0.001},
                {"name": "stim", "amp": 0.025, "duration": 0.001},
            ],
        )
        config = base_config(protocol)

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "timed_random_stimulation.py",
                "--protocol",
                "protocol",
                "--mock",
                "--max-events",
                "3",
                "--session-id",
                "smoke_session",
                "--log-dir",
                temp_dir,
                "--seed",
                "7",
            ]
            with mock.patch.object(timed_random, "CONFIG", config), mock.patch.object(
                timed_random, "HW_CONF", config["hardware"]
            ), mock.patch.object(sys, "argv", argv):
                self.assertEqual(timed_random.main(), 0)

            session_dir = Path(temp_dir) / "smoke_session"
            metadata_path = session_dir / "session_metadata.json"
            csv_path = session_dir / "stim_events.csv"

            self.assertTrue(metadata_path.exists())
            self.assertTrue(csv_path.exists())

            with csv_path.open(newline="", encoding="utf-8") as file:
                events = list(csv.DictReader(file))
            event_names = [row["event"] for row in events]

            self.assertIn("session_start", event_names)
            self.assertIn("session_end", event_names)
            self.assertIn("stim_sham", event_names)
            self.assertIn("stim_on", event_names)
            self.assertIn("stim_off", event_names)

    def test_notify_pycontrol_sends_marker_codes(self):
        sent_codes = []

        class FakePyControlEventLink:
            def __init__(self, port, baud_rate=9600, timeout=0.01):
                self.port = port
                self.baud_rate = baud_rate
                self.timeout = timeout

            def open(self):
                return self

            def send_code(self, code):
                sent_codes.append(code)

            def close(self):
                sent_codes.append("closed")

        protocol = valid_protocol(
            interval_s=0.005,
            conditions=[
                {"name": "sham", "sham": True, "amp": [0, 0], "duration": 0.001},
                {"name": "stim", "amp": 0.025, "duration": 0.001},
            ],
        )
        config = base_config(protocol)

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "timed_random_stimulation.py",
                "--protocol",
                "protocol",
                "--mock",
                "--notify-pycontrol",
                "--pycontrol-port",
                "MOCK_PYCONTROL",
                "--max-events",
                "2",
                "--session-id",
                "notify_session",
                "--log-dir",
                temp_dir,
                "--seed",
                "7",
            ]
            with mock.patch.object(timed_random, "CONFIG", config), mock.patch.object(
                timed_random, "HW_CONF", config["hardware"]
            ), mock.patch.object(
                timed_random, "PyControlEventLink", FakePyControlEventLink
            ), mock.patch.object(sys, "argv", argv):
                self.assertEqual(timed_random.main(), 0)

        self.assertEqual(sent_codes, [110, 104, 101, 102, 111, "closed"])


if __name__ == "__main__":
    unittest.main()
