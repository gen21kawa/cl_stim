def configured_ports(config, list_key, fallback_key):
    """Return serial port candidates from a list config or single fallback."""
    if list_key in config:
        raw_ports = config[list_key]
    else:
        raw_ports = config.get(fallback_key)

    if raw_ports is None:
        return []
    if isinstance(raw_ports, str):
        return [raw_ports]
    return [str(port) for port in raw_ports]


def available_serial_ports():
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed in this Python environment. "
            "Use the conda environment declared in env.yml or install pyserial."
        ) from exc

    return [port.device for port in list_ports.comports()]


def resolve_serial_port(
    config,
    list_key,
    fallback_key,
    *,
    label="serial port",
    require_available=False,
):
    candidates = configured_ports(config, list_key, fallback_key)
    if not candidates:
        raise RuntimeError(
            f"No {label} configured. Set {list_key} or {fallback_key} in config.toml."
        )

    if not require_available:
        return candidates[0], candidates, None

    available = available_serial_ports()
    available_by_name = {port.upper(): port for port in available}
    for candidate in candidates:
        if candidate.upper() in available_by_name:
            return available_by_name[candidate.upper()], candidates, available

    available_msg = ", ".join(available) if available else "(none detected)"
    checked_msg = ", ".join(candidates)
    raise RuntimeError(
        f"No configured {label} is available. Checked: {checked_msg}. "
        f"Detected serial ports: {available_msg}."
    )
