## How To Use

This repo has three normal ways to run stimulation:

- `manual_stimulation.py`: the operator types commands at a `stim>` prompt.
- `passive_stimulation.py`: pyControl owns the trial timing and sends condition
  codes over UART. This is a role-named wrapper around `run_stimulation.py`.
- `timed_random_stimulation.py`: the stim computer owns timing and chooses
  fixed-interval train-stimulation conditions from a balanced shuffled list.

Use mock mode first on any new setup:

```bash
python manual_stimulation.py --mock --profile m1_mapping_low
python passive_stimulation.py --mock --experiment m1_mapping_low
python timed_random_stimulation.py --mock --protocol m1_random_timed --max-events 6
```

Use real hardware only after the channel map, stimulator ports, and profile
limits in `config.toml` have been checked:

```bash
python manual_stimulation.py --animal M111 --profile m1_mapping_low --real
python passive_stimulation.py --animal M111 --experiment m1_mapping_low --real
python timed_random_stimulation.py --animal M111 --protocol m1_random_timed --real
```

For task/passive runs, `--real` also requires the selected experiment to have
`real_enabled = true` in `config.toml`. For timed-random runs, the selected
timed-random protocol must have `real_enabled = true`.

## Manual Stimulation

Manual mode is useful for bench tests, threshold checks, and operator-triggered
stimulation during a session:

```bash
python manual_stimulation.py --animal M111 --profile m1_mapping_low --real
```

At the `stim>` prompt:

```text
amp                   # show current amplitude
amp 0.03              # set current amplitude in mA
pulse                 # pulse using the default duration and current amplitude
pulse 0.2             # 0.2 s pulse at the current amplitude
pulse 0.2 0.03        # 0.2 s pulse at 0.03 mA
pulse amp 0.03        # default-duration pulse at 0.03 mA
on 0.03               # start train stimulation at 0.03 mA
off                   # stop train stimulation
status                # show mode, profile, pulse duration, and current amp
quit                  # stop and exit
```

The profile `amp` is the session maximum. Manual commands may use any finite,
nonnegative amplitude at or below that maximum. For multi-channel profiles, pass
one amplitude per configured channel, for example:

```text
amp 0.03 0.04
pulse 0.2 0.03 0.04
```

Manual amplitude changes are allowed between stimulation events. If train
stimulation is already on, turn it off before changing amplitude or sending a
new pulse.

## Timed Random Stimulation

Timed-random mode is useful when the stimulation computer should deliver trains
at a fixed cadence without pyControl triggers:

```bash
python timed_random_stimulation.py --mock --protocol m1_random_timed --max-events 6
```

Protocols are configured under `[timed_random_protocols.<name>]` in
`config.toml`. Each protocol points at a train-mode stimulation profile for
frequency, pulse width, channels, inter-phase, and amplitude safety limits. The
protocol adds an onset-to-onset `interval_s` and a list of conditions with
`name`, `amp`, and `duration`.

The runner uses balanced shuffle randomization: all configured conditions are
used once in random order, then the deck is reshuffled. Pass `--seed` to repeat a
specific order; otherwise a seed is generated and saved in
`session_metadata.json`.

Sham can be marked with `sham = true` or by setting all amplitudes to zero. Sham
events are logged as `stim_sham` and no stimulation command is sent. Non-sham
events log `stim_on_request`, `stim_on`, and `stim_off`.

For a real session, first set the protocol's `real_enabled = true` after
checking the channel map, ports, and amplitude limits:

```bash
python timed_random_stimulation.py --animal M111 --protocol m1_random_timed --real
```

To let pyControl log the timed-random events, start the pyControl task's
`hw.bci_link` and add `--notify-pycontrol`. The stimulation computer still owns
the timing; pyControl only receives marker codes:

```bash
python timed_random_stimulation.py --animal M111 --protocol m2_random_timed --real --notify-pycontrol
```

## Task Passive Conditions

Passive/task mode waits for pyControl to send a 2-byte little-endian integer
condition code over the trigger UART. Start the stim computer server before
starting the pyControl task:

```bash
python passive_stimulation.py --animal M111 --experiment m1_mapping_low --real
```

`passive_stimulation.py` and `run_stimulation.py` use the same implementation,
so this is equivalent:

```bash
python run_stimulation.py --animal M111 --experiment m1_mapping_low --real
```

Conditions are configured under `[experiments.<name>.commands]` in
`config.toml`. The current `m1_mapping_low` example is:

```text
1: sham       pw_fraction=[0, 0]
2: left_m1    pw_fraction=[1, 0]
3: right_m1   pw_fraction=[0, 1]
4: bilateral  pw_fraction=[1, 1]
```

`pw_fraction` scales each configured channel's profile pulse width. A condition
with all zeros is sham: it is logged as `stim_sham` and no stimulation command is
sent. In train mode, each condition uses the profile `duration` unless that
command has its own `duration` field. In single-pulse mode, each condition sends
one run-once pulse.

Task/passive mode uses the profile amplitude from `config.toml` for all
conditions. To use a different amplitude ceiling, change or add a stimulation
profile and point the experiment at it before starting the server.

Command code `0` is reserved as a session marker. Use positive integer condition
codes for stimulation/sham conditions.

On the pyControl side, choose the passive condition in the task and send the
matching integer. For example, with `m1_mapping_low`, send `1` for sham, `2` for
left M1, `3` for right M1, or `4` for bilateral. The stimulation server logs the
received code, the resolved condition name, the pulse width values, amplitude,
and on/off timing.

## Logging

Manual, passive, and timed-random stimulation runs write a local stim-computer log
under `logs/stim_events/<session_id>/` by default. For real sessions, pass an
animal ID so logs land beside behavior/ephys data. With `--animal M111`, the
scripts create or reuse:

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

## pyControl Markers

pyControl notification is optional and disabled by default in all host-driven scripts.
Enable it when the pyControl task has started `hw.bci_link`:

```bash
python manual_stimulation.py --animal M111 --profile m1_mapping_low --real --notify-pycontrol
python passive_stimulation.py --animal M111 --experiment m1_mapping_low --real --notify-pycontrol
python timed_random_stimulation.py --animal M111 --protocol m2_random_timed --real --notify-pycontrol
```

With `--notify-pycontrol`, the stimulation process echoes `stim_on`/`stim_off`
or `stim_pulse` markers back over the UART link so the pyControl task can track
real stim timing, not just the moment it sent the trigger. Timed-random mode
also sends `stim_sham` for sham conditions. The host sends the same 2-byte
little-endian integer markers expected by
`../TreadmillTasks/devices/UARTlink.py`.

The companion pyControl task that triggers + tracks stimulation is
`../TreadmillTasks/tasks/6-run-to-stim-electrical-spontaneous.py` (based on the
task-5 spontaneous + penalty + auto-delivery template). It sends a command code
(`v.stim_command_code`) at the rewarded outcome and logs markers received back.

Default marker codes are configured in `config.toml`:

- `101`: `stim_on`
- `102`: `stim_off`
- `103`: `stim_pulse`
- `104`: `stim_sham`
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
        elif code == 104:
            print("{}, external_stim_sham code=104".format(get_current_time()))
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
