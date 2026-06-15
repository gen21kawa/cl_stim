"""Shared helpers for the stimulation entry points.

`run_stimulation.py` (UART-triggered server) and `manual_stimulation.py`
(interactive operator control) have separate main loops but share the same
pulse-mode normalization, per-event logging schema, and session-log setup. Those
pieces live here so both scripts stay in sync.
"""

from utils.loader import CONFIG
from utils.stim_event_log import (
    StimEventLogger,
    resolve_channel_map,
    resolve_output_dir,
    resolve_session_dir,
)


def normalize_pulse_mode(stim_profile, *, validate=False):
    """Normalize a profile's pulse mode to the names used by MatlabStimulator.

    With ``validate=True`` an unrecognized value raises ``ValueError`` (used by
    the trigger server, which fails fast on bad config). Without it, the value
    is passed through after the ``single`` -> ``single_pulse`` alias (the
    interactive script is more lenient).
    """
    pulse_mode = str(stim_profile.get("pulse_mode", "train")).lower()
    if pulse_mode == "single":
        pulse_mode = "single_pulse"
    if validate and pulse_mode not in ("train", "single_pulse"):
        raise ValueError("pulse_mode must be 'train' or 'single_pulse'.")
    return pulse_mode


def build_event_fields(
    source,
    profile_name,
    stim_profile,
    stim_port,
    pulse_mode,
    mock_mode,
    *,
    experiment=None,
    command=None,
    command_code=None,
    pycontrol_code=None,
    amp_values=None,
    pw_values=None,
    duration=None,
    status=None,
):
    """Build the shared columns written for each stimulation event row.

    Keeping the common stimulation metadata in one helper makes each event row
    comparable across scripts and event types. ``None`` values are dropped so
    sparse CSV columns stay empty rather than literal "None".
    """
    channel_metadata = stim_profile.get("_channel_metadata", {})
    fields = {
        "source": source,
        "experiment": experiment,
        "profile": profile_name,
        "pulse_mode": pulse_mode,
        "channels": stim_profile.get("channel", [1]),
        "channel_labels": channel_metadata.get("channel_labels"),
        "physical_contacts": channel_metadata.get("physical_contacts"),
        "freq_hz": stim_profile.get("freq"),
        "pw_ms": pw_values if pw_values is not None else stim_profile.get("pw"),
        "amp_mA": amp_values if amp_values is not None else stim_profile.get("amp"),
        "duration_s": duration,
        "inter_phase_s": stim_profile.get("inter_phase"),
        "stim_port": stim_port,
        "mock_mode": mock_mode,
        "command": command,
        "command_code": command_code,
        "pycontrol_code": pycontrol_code,
        "status": status,
    }
    return {key: value for key, value in fields.items() if value is not None}


def resolve_channel_metadata(stim_profile, *, print_warnings=True):
    """Resolve and attach channel-map metadata onto ``stim_profile`` in place.

    Returns the metadata dict. The metadata is stored under ``_channel_metadata``
    so :func:`build_event_fields` can read channel labels/contacts for each row.
    """
    channel_metadata = resolve_channel_map(
        stim_profile.get("channel"), CONFIG.get("channel_map", {})
    )
    stim_profile["_channel_metadata"] = channel_metadata
    if print_warnings:
        for warning in channel_metadata["warnings"]:
            print(f"!! Channel map warning: {warning}")
    return channel_metadata


def make_event_logger(
    *,
    script,
    animal,
    session_id,
    data_root,
    log_dir,
    root_dir,
    profile_name,
    stim_profile,
    channel_metadata,
    stim_port,
    mock_mode,
    extra=None,
    enabled=True,
):
    """Construct a :class:`StimEventLogger` with the standard metadata payload.

    ``extra`` holds script-specific metadata keys (e.g. the experiment/command
    map for the trigger server, or the pyControl event config for manual mode).
    """
    log_conf = CONFIG.get("logging", {})
    session_dir, resolved_session_id = resolve_session_dir(
        animal,
        session_id,
        data_root or log_conf.get("data_root", "data"),
        root_dir,
    )
    resolved_log_dir = resolve_output_dir(
        log_dir or log_conf.get("stim_event_dir", "logs/stim_events"),
        root_dir,
    )

    metadata = {
        "script": script,
        "animal": animal,
        "session_dir": session_dir,
        "profile_name": profile_name,
        "stim_profile": stim_profile,
        "channel_map": channel_metadata["channel_map"],
        "channel_map_warnings": channel_metadata["warnings"],
        "stim_port": stim_port,
        "mock_mode": mock_mode,
    }
    if extra:
        metadata.update(extra)

    return StimEventLogger(
        resolved_log_dir,
        session_id=resolved_session_id,
        session_dir=session_dir,
        metadata=metadata,
        enabled=enabled,
    )
