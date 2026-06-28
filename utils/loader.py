import copy
import os
import sys

# Compat for Python < 3.11
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_toml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "rb") as file:
        return tomllib.load(file)


def merge_config(base, override):
    """Recursively merge two TOML config dictionaries."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config_path(path, *, root_dir):
    if os.path.isabs(path):
        return path
    return os.path.join(root_dir, path)


def load_config(config_path=None):
    if config_path is None:
        root_dir = repo_root()
        config_path = os.path.join(root_dir, "config.toml")
    else:
        config_path = os.path.abspath(config_path)
        root_dir = os.path.dirname(config_path)

    config = read_toml(config_path)
    experiment_config = config.get("run", {}).get("experiment_config")
    if experiment_config:
        experiment_path = resolve_config_path(experiment_config, root_dir=root_dir)
        experiment = read_toml(experiment_path)
        config = merge_config(config, experiment)
        config.setdefault("run", {})["loaded_experiment_config"] = experiment_config

    return config

CONFIG = load_config()
