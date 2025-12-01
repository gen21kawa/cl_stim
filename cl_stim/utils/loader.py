import os
import sys

# Compat for Python < 3.11
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

def load_config():
    # Find the config.toml file relative to this script
    # Since loader.py is now in the root, config.toml is in the same directory.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, "config.toml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    return config

CONFIG = load_config()