# Standard Libraries
from pathlib import Path
import yaml


def load_config(file_path: Path) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        file_path (Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data