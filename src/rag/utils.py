import yaml
from pathlib import Path


def load_config(file_path: Path) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        file_path (Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data