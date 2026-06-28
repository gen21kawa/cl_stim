import tempfile
import unittest
from pathlib import Path

from utils import loader


class ConfigLoaderTests(unittest.TestCase):
    def test_experiment_config_is_merged_over_base_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configs = root / "configs"
            configs.mkdir()
            (root / "config.toml").write_text(
                """
[run]
mock_mode = true
experiment_config = "configs/example.toml"

[hardware]
stimulator_port = "COM9"
baud_rate = 9600

[logging]
stim_events_enabled = true
""".strip(),
                encoding="utf-8",
            )
            (configs / "example.toml").write_text(
                """
[run]
active_movement_triggered_protocol = "m2_movement_triggered"

[hardware]
baud_rate = 115200

[experiment]
type = "movement_triggered"
name = "m2_movement_triggered"
""".strip(),
                encoding="utf-8",
            )

            config = loader.load_config(str(root / "config.toml"))

        self.assertTrue(config["run"]["mock_mode"])
        self.assertEqual(config["run"]["loaded_experiment_config"], "configs/example.toml")
        self.assertEqual(
            config["run"]["active_movement_triggered_protocol"],
            "m2_movement_triggered",
        )
        self.assertEqual(config["hardware"]["stimulator_port"], "COM9")
        self.assertEqual(config["hardware"]["baud_rate"], 115200)
        self.assertTrue(config["logging"]["stim_events_enabled"])

    def test_missing_experiment_config_fails_fast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.toml"
            path.write_text(
                """
[run]
experiment_config = "configs/missing.toml"
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                loader.load_config(str(path))


if __name__ == "__main__":
    unittest.main()
