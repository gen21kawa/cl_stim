## Stimulation Timing and Behavior Logging

Manual and pyControl-triggered stimulation runs write a local stim-computer log
under `logs/stim_events/<session_id>/` by default. For real sessions, pass an
animal ID so logs land beside behavior/ephys data:

```bash
python manual_stimulation.py --animal M111 --profile m1_mapping_low --real --notify-pycontrol
python run_stimulation.py --animal M111 --experiment m1_mapping_low --real
```

With `--animal M111`, the scripts create or reuse:

```text
<data_root>/M111/M111_YYYY_MM_DD_HH_MM/
```

Use `--session-id M111_2026_06_11_14_00` to attach to a specific session and
`--data-root PATH` to override `[logging].data_root`.

- `session_metadata.json`: script, profile, command map, ports, and config snapshot.
- `stim_events.csv`: one row per trigger, stim onset command, stop command, sham,
  session marker, or error.

The CSV includes `channels`, `channel_labels`, and `physical_contacts`. These
come from `[channel_map]` in `config.toml`; update the placeholder entries after
checking the stimulator/electrode documentation. Missing channel-map entries log
as `unknown` and print a warning rather than stopping the session.

For manual stimulation, pyControl notification is optional and disabled by
default. Enable it when the pyControl task has started `hw.bci_link`:

```bash
python manual_stimulation.py --animal M111 --profile m1_mapping_low --real --notify-pycontrol
```

The host sends the same 2-byte little-endian integer markers expected by
`../TreadmillTasks/devices/UARTlink.py`.

Default marker codes are configured in `config.toml`:

- `101`: `stim_on`
- `102`: `stim_off`
- `103`: `stim_pulse`
- `110`: `session_start`
- `111`: `session_end`

On the pyControl side, use the TreadmillTasks pattern:

```python
events = [
    # existing task events...
    "cursor_update",
]

def run_start():
    hw.bci_link.start()

def run_end():
    hw.bci_link.stop()

def all_states(event):
    if event == "cursor_update":
        code = hw.bci_link.spk
        if code == 101:
            print("{}, external_stim_on code=101".format(get_current_time()))
        elif code == 102:
            print("{}, external_stim_off code=102".format(get_current_time()))
        elif code == 103:
            print("{}, external_stim_pulse code=103".format(get_current_time()))
        elif code == 110:
            print("{}, external_stim_session_start code=110".format(get_current_time()))
        elif code == 111:
            print("{}, external_stim_session_end code=111".format(get_current_time()))
```

`UARTlink` emits an event only when the received integer changes, so keep
separate on/off/session/pulse codes and avoid using one repeated code for every
manual event.

Best practice for high-precision timing is still a hardware TTL/sync pulse into
pyControl and the ephys acquisition system, with the local CSV/JSON logs used
for parameters and auditability. The serial markers here are a practical
software marker and should be validated against TTL timing before being treated
as the ground-truth stimulation onset.
