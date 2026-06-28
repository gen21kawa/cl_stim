import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import movement_triggered_stimulation as movement_stim


def base_config(protocol):
    return {
        "run": {
            "mock_mode": True,
            "active_movement_triggered_protocol": "protocol",
        },
        "hardware": {
            "stimulator_port": "MOCK_STIM",
            "trigger_port": "MOCK_TRIGGER",
            "baud_rate": 9600,
        },
        "logging": {"stim_events_enabled": True},
        "pycontrol_events": {
            "enabled": False,
            "port": "MOCK_TRIGGER",
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
        "movement_triggered_protocols": {"protocol": protocol},
    }


def valid_protocol(**updates):
    protocol = {
        "profile": "train_profile",
        "duration": 0.001,
        "trigger_code": 1,
        "real_enabled": False,
        "conditions": [
            {"name": "sham", "sham": True, "amp": [0, 0]},
            {"name": "stim", "amp": 0.025},
        ],
    }
    protocol.update(updates)
    return protocol


def self_contained_config(**updates):
    config = {
        "run": {"mock_mode": True},
        "hardware": {
            "stimulator_port": "MOCK_STIM",
            "trigger_port": "MOCK_TRIGGER",
            "baud_rate": 9600,
        },
        "logging": {"stim_events_enabled": True},
        "pycontrol_events": {
            "enabled": False,
            "port": "MOCK_TRIGGER",
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
        "experiment": {
            "type": "movement_triggered",
            "name": "m2_movement_triggered",
            "real_enabled": False,
            "trigger_code": 1,
            "duration": 0.001,
        },
        "stim": {
            "freq": 50,
            "max_amp": [0.05, 0.06],
            "pw": 0.2,
            "pulse_mode": "train",
            "channel": [1, 2],
            "inter_phase": 50e-6,
        },
        "conditions": [
            {"name": "sham", "sham": True, "amp": [0, 0]},
            {"name": "stim", "amp": 0.025},
        ],
    }
    config.update(updates)
    return config


def validate_with(protocol):
    config = base_config(protocol)
    with mock.patch.object(movement_stim, "CONFIG", config):
        return movement_stim.validate_movement_triggered_protocol("protocol", protocol)


class MovementTriggeredValidationTests(unittest.TestCase):
    def test_self_contained_config_resolves_and_validates(self):
        config = self_contained_config()
        with mock.patch.object(movement_stim, "CONFIG", config):
            protocol_name, protocol = movement_stim.resolve_protocol()
            validated = movement_stim.validate_movement_triggered_protocol(
                protocol_name, protocol
            )

        self.assertEqual(protocol_name, "m2_movement_triggered")
        self.assertEqual(validated["profile_name"], "m2_movement_triggered")
        self.assertEqual(validated["stim_profile"]["amp"], [0.05, 0.06])
        self.assertEqual(validated["conditions"][1].amp_values, [0.025, 0.025])

    def test_self_contained_config_rejects_protocol_mismatch(self):
        config = self_contained_config()
        with mock.patch.object(movement_stim, "CONFIG", config):
            with self.assertRaisesRegex(ValueError, "requested 'other_protocol'"):
                movement_stim.resolve_protocol("other_protocol")

    def test_self_contained_amp_above_max_fails(self):
        config = self_contained_config(
            conditions=[{"name": "too_high", "amp": [0.05, 0.061]}]
        )
        with mock.patch.object(movement_stim, "CONFIG", config):
            protocol_name, protocol = movement_stim.resolve_protocol()
            with self.assertRaisesRegex(ValueError, "exceeds the profile maximum"):
                movement_stim.validate_movement_triggered_protocol(
                    protocol_name, protocol
                )

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

    def test_duration_must_be_positive(self):
        protocol = valid_protocol(duration=0)
        with self.assertRaisesRegex(ValueError, "duration"):
            validate_with(protocol)

    def test_trigger_code_zero_is_reserved(self):
        protocol = valid_protocol(trigger_code=0)
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_with(protocol)

    def test_amp_above_profile_max_fails(self):
        protocol = valid_protocol(
            conditions=[{"name": "too_high", "amp": [0.05, 0.061]}]
        )
        with self.assertRaisesRegex(ValueError, "exceeds the profile maximum"):
            validate_with(protocol)

    def test_scalar_per_channel_expansion_and_sham_detection(self):
        protocol = valid_protocol(
            conditions=[
                {"name": "scalar", "amp": 0.025},
                {"name": "per_channel", "amp": [0.01, 0.02]},
                {"name": "zero_sham", "amp": [0, 0]},
                {"name": "flagged_sham", "sham": True, "amp": [0.01, 0.02]},
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


class MovementTriggeredShuffleTests(unittest.TestCase):
    def test_balanced_shuffle_reproducible_with_seed(self):
        first = movement_stim.BalancedConditionDeck(3, random.Random(123))
        second = movement_stim.BalancedConditionDeck(3, random.Random(123))

        first_draws = [first.next_index() for _ in range(6)]
        second_draws = [second.next_index() for _ in range(6)]

        self.assertEqual(first_draws, second_draws)
        self.assertEqual(sorted(index for index, cycle in first_draws[:3]), [0, 1, 2])
        self.assertEqual(sorted(index for index, cycle in first_draws[3:]), [0, 1, 2])
        self.assertEqual({cycle for index, cycle in first_draws[:3]}, {1})
        self.assertEqual({cycle for index, cycle in first_draws[3:]}, {2})


class FakeSerial:
    def __init__(self, codes):
        self._input = bytearray()
        for code in codes:
            self._input.extend(int(code).to_bytes(2, byteorder="little", signed=False))
        self.written_codes = []
        self.closed = False
        self.reset_count = 0

    @property
    def in_waiting(self):
        return len(self._input)

    def read(self, count):
        data = bytes(self._input[:count])
        del self._input[:count]
        return data

    def write(self, data):
        self.written_codes.append(int.from_bytes(data, byteorder="little", signed=False))
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self.reset_count += 1
        self._input.clear()

    def close(self):
        self.closed = True


class FakeStimulator:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        FakeStimulator.instances.append(self)

    @staticmethod
    def _normalize_channels(channels):
        if channels is None:
            channels = [1]
        elif isinstance(channels, int):
            channels = [channels]
        normalized = tuple(int(channel) for channel in channels)
        if not normalized:
            raise ValueError("At least one stimulation channel is required.")
        return normalized

    def configure(self, **kwargs):
        self.calls.append(("configure", kwargs))

    def connect(self):
        self.calls.append(("connect", None))

    def stimulate(self, **kwargs):
        self.calls.append(("stimulate", kwargs))

    def stop(self):
        self.calls.append(("stop", None))

    def close(self):
        self.calls.append(("close", None))


class MovementTriggeredSmokeTests(unittest.TestCase):
    def setUp(self):
        FakeStimulator.instances = []

    def run_smoke_config(self, config, trigger_codes=None, protocol_arg=None):
        fake_link = FakeSerial(trigger_codes or [1])

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "movement_triggered_stimulation.py",
                "--mock",
                "--max-triggers",
                "1",
                "--session-id",
                "movement_session",
                "--log-dir",
                temp_dir,
                "--seed",
                "7",
                "--quiet",
            ]
            if protocol_arg is not None:
                argv[1:1] = ["--protocol", protocol_arg]

            with mock.patch.object(movement_stim, "CONFIG", config), mock.patch.object(
                movement_stim, "HW_CONF", config["hardware"]
            ), mock.patch.object(
                movement_stim, "MatlabStimulator", FakeStimulator
            ), mock.patch.object(
                movement_stim, "open_trigger_link", return_value=fake_link
            ), mock.patch.object(
                sys, "argv", argv
            ):
                self.assertEqual(movement_stim.main(), 0)

            csv_path = Path(temp_dir) / "movement_session" / "stim_events.csv"
            with csv_path.open(newline="", encoding="utf-8") as file:
                events = list(csv.DictReader(file))
        return fake_link, FakeStimulator.instances[0], events

    def run_smoke(self, protocol):
        config = base_config(protocol)
        fake_link = FakeSerial([1])

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "movement_triggered_stimulation.py",
                "--protocol",
                "protocol",
                "--mock",
                "--max-triggers",
                "1",
                "--session-id",
                "movement_session",
                "--log-dir",
                temp_dir,
                "--seed",
                "7",
                "--quiet",
            ]
            with mock.patch.object(movement_stim, "CONFIG", config), mock.patch.object(
                movement_stim, "HW_CONF", config["hardware"]
            ), mock.patch.object(
                movement_stim, "MatlabStimulator", FakeStimulator
            ), mock.patch.object(
                movement_stim, "open_trigger_link", return_value=fake_link
            ), mock.patch.object(
                sys, "argv", argv
            ):
                self.assertEqual(movement_stim.main(), 0)

            csv_path = Path(temp_dir) / "movement_session" / "stim_events.csv"
            with csv_path.open(newline="", encoding="utf-8") as file:
                events = list(csv.DictReader(file))
        return fake_link, FakeStimulator.instances[0], events

    def test_self_contained_mock_stim_trigger_writes_on_and_off_markers(self):
        config = self_contained_config(conditions=[{"name": "stim", "amp": 0.025}])

        fake_link, fake_stim, events = self.run_smoke_config(config)

        event_names = [row["event"] for row in events]
        self.assertIn("trigger_received", event_names)
        self.assertIn("stim_on_request", event_names)
        self.assertIn("stim_on", event_names)
        self.assertIn("stim_off", event_names)
        self.assertEqual(fake_link.written_codes, [101, 102])
        self.assertIn("stimulate", [name for name, kwargs in fake_stim.calls])

    def test_mock_stim_trigger_writes_on_and_off_markers(self):
        protocol = valid_protocol(conditions=[{"name": "stim", "amp": 0.025}])

        fake_link, fake_stim, events = self.run_smoke(protocol)

        event_names = [row["event"] for row in events]
        self.assertIn("trigger_received", event_names)
        self.assertIn("stim_on_request", event_names)
        self.assertIn("stim_on", event_names)
        self.assertIn("stim_off", event_names)
        self.assertEqual(fake_link.written_codes, [101, 102])
        self.assertIn("stimulate", [name for name, kwargs in fake_stim.calls])

    def test_mock_sham_trigger_writes_sham_and_off_markers(self):
        protocol = valid_protocol(conditions=[{"name": "sham", "sham": True, "amp": 0}])

        fake_link, fake_stim, events = self.run_smoke(protocol)

        event_names = [row["event"] for row in events]
        self.assertIn("trigger_received", event_names)
        self.assertIn("stim_sham", event_names)
        self.assertIn("stim_off", event_names)
        self.assertEqual(fake_link.written_codes, [104, 102])
        self.assertNotIn("stimulate", [name for name, kwargs in fake_stim.calls])

    def test_unknown_trigger_is_logged_and_ignored(self):
        protocol = valid_protocol(conditions=[{"name": "stim", "amp": 0.025}])
        config = base_config(protocol)
        fake_link = FakeSerial([99])

        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "movement_triggered_stimulation.py",
                "--protocol",
                "protocol",
                "--mock",
                "--runtime",
                "0.05",
                "--session-id",
                "unknown_session",
                "--log-dir",
                temp_dir,
                "--quiet",
            ]
            with mock.patch.object(movement_stim, "CONFIG", config), mock.patch.object(
                movement_stim, "HW_CONF", config["hardware"]
            ), mock.patch.object(
                movement_stim, "MatlabStimulator", FakeStimulator
            ), mock.patch.object(
                movement_stim, "open_trigger_link", return_value=fake_link
            ), mock.patch.object(
                sys, "argv", argv
            ):
                self.assertEqual(movement_stim.main(), 0)

            csv_path = Path(temp_dir) / "unknown_session" / "stim_events.csv"
            with csv_path.open(newline="", encoding="utf-8") as file:
                events = list(csv.DictReader(file))

        self.assertIn("trigger_unknown", [row["event"] for row in events])
        self.assertEqual(fake_link.written_codes, [])


if __name__ == "__main__":
    unittest.main()
