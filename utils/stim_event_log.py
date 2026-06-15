"""Structured stimulation event logging."""

import csv
import datetime as dt
import json
import time
from pathlib import Path


FIELDNAMES = [
    "event_index",
    "session_id",
    "wall_time_iso",
    "time_unix_s",
    "perf_counter_s",
    "perf_counter_ns",
    "event",
    "source",
    "experiment",
    "profile",
    "command",
    "command_code",
    "pycontrol_code",
    "pulse_mode",
    "channels",
    "channel_labels",
    "physical_contacts",
    "freq_hz",
    "pw_ms",
    "amp_mA",
    "duration_s",
    "inter_phase_s",
    "stim_port",
    "mock_mode",
    "status",
    "error",
    "details_json",
]


def default_session_id(prefix="stim"):
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def round_dt_to_nearest_10min(now=None):
    """Round a datetime to the nearest 10-minute mark (00, 10, 20, ...).

    Uses timedelta arithmetic so hour/day rollover is handled (e.g. 13:57 ->
    14:00, 14:16 -> 14:20). Seconds and microseconds are dropped.
    """
    if now is None:
        now = dt.datetime.now()
    floor = now.replace(minute=(now.minute // 10) * 10, second=0, microsecond=0)
    remainder = now - floor
    if remainder >= dt.timedelta(minutes=5):
        floor += dt.timedelta(minutes=10)
    return floor


def default_animal_session_id(animal, now=None):
    rounded = round_dt_to_nearest_10min(now)
    return f"{animal}_{rounded.strftime('%Y_%m_%d_%H_%M')}"


def resolve_output_dir(path, root_dir):
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = Path(root_dir) / out
    return out


def resolve_session_dir(animal, session_id, data_root, root_dir):
    """Return the behavior/ephys session folder for animal-scoped logging."""
    if not animal:
        return None, session_id

    resolved_session_id = session_id or default_animal_session_id(animal)
    out = resolve_output_dir(data_root, root_dir)
    return out / animal / resolved_session_id, resolved_session_id


def normalize_channels(channels):
    if channels is None:
        return [1]
    if isinstance(channels, int):
        return [channels]
    return [int(channel) for channel in channels]


def resolve_channel_map(channels, channel_map_config):
    """Resolve stimulator channel numbers to loggable physical metadata."""
    resolved = []
    labels = []
    physical_contacts = []
    warnings = []

    for channel in normalize_channels(channels):
        mapping = channel_map_config.get(str(channel), {})
        if not mapping:
            mapping = {
                "label": "unknown",
                "physical_contact": "unknown",
                "target": "unknown",
                "hemisphere": "unknown",
                "notes": "No channel_map entry configured.",
            }
            warnings.append(f"channel {channel} missing from [channel_map]")

        entry = {"channel": channel, **_json_safe(mapping)}
        entry.setdefault("label", "unknown")
        entry.setdefault("physical_contact", "unknown")
        resolved.append(entry)
        labels.append(entry["label"] or "unknown")
        physical_contacts.append(entry["physical_contact"] or "unknown")

    return {
        "channels": normalize_channels(channels),
        "channel_labels": labels,
        "physical_contacts": physical_contacts,
        "channel_map": resolved,
        "warnings": warnings,
    }


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _cell(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if value is None:
        return ""
    return value


class StimEventLogger:
    """Write one session directory with metadata JSON and event CSV."""

    def __init__(
        self,
        output_dir=None,
        session_id=None,
        metadata=None,
        enabled=True,
        session_dir=None,
    ):
        self.enabled = enabled
        self.session_id = session_id or default_session_id()
        self.event_index = 0
        self.session_dir = None
        self.log_dir = None
        self.csv_path = None
        self.metadata_path = None
        self._file = None
        self._writer = None

        if not self.enabled:
            return

        if session_dir is not None:
            self.session_dir = Path(session_dir)
            self.log_dir = self.session_dir
        else:
            self.log_dir = Path(output_dir) / self.session_id
            self.session_dir = self.log_dir

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "stim_events.csv"
        self.metadata_path = self.log_dir / "session_metadata.json"
        csv_exists = self.csv_path.exists()

        metadata_payload = {
            "session_id": self.session_id,
            "created_wall_time_iso": dt.datetime.now().astimezone().isoformat(),
            "session_dir": str(self.session_dir),
            "log_dir": str(self.log_dir),
            "metadata": _json_safe(metadata or {}),
        }
        self.metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if csv_exists:
            with self.csv_path.open("r", newline="", encoding="utf-8") as existing:
                for row in csv.DictReader(existing):
                    try:
                        self.event_index = max(
                            self.event_index, int(row.get("event_index", 0))
                        )
                    except (TypeError, ValueError):
                        pass

        self._file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if not csv_exists:
            self._writer.writeheader()
        self._file.flush()

    def record(self, event, **fields):
        if not self.enabled:
            return None

        self.event_index += 1
        now = time.time()
        row = {
            "event_index": self.event_index,
            "session_id": self.session_id,
            "wall_time_iso": dt.datetime.now().astimezone().isoformat(),
            "time_unix_s": f"{now:.6f}",
            "perf_counter_s": f"{time.perf_counter():.9f}",
            "perf_counter_ns": time.perf_counter_ns(),
            "event": event,
        }

        details = {}
        for key, value in fields.items():
            if key in FIELDNAMES:
                row[key] = _cell(value)
            else:
                details[key] = _json_safe(value)
        row["details_json"] = json.dumps(details, sort_keys=True) if details else ""

        complete_row = {key: row.get(key, "") for key in FIELDNAMES}
        self._writer.writerow(complete_row)
        self._file.flush()
        return complete_row

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
