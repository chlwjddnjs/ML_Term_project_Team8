"""Configuration loading.  Owner: shared."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path="config.yaml", data_profile=None):
    """Read the YAML config and resolve the selected data profile.

    Args:
        path: path to the YAML file (relative to the project root or absolute).
        data_profile: overrides the `data` key (e.g. "sample" for a smoke test).

    Returns:
        dict with every key of the YAML file plus `paths`, the resolved
        (absolute) file paths of the selected profile.
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profile_name = data_profile or cfg["data"]
    profile = cfg["data_profiles"][profile_name]
    cfg["data"] = profile_name
    cfg["min_user_ratings"] = profile["min_user_ratings"]
    cfg["paths"] = {
        "xwines_wines": PROJECT_ROOT / profile["xwines_wines"],
        "xwines_ratings": PROJECT_ROOT / profile["xwines_ratings"],
        "winemag": PROJECT_ROOT / cfg["winemag_path"],
        "output_dir": PROJECT_ROOT / cfg["output_dir"],
    }
    cfg["paths"]["output_dir"].mkdir(parents=True, exist_ok=True)
    return cfg
