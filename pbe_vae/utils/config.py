"""
pbe_vae.utils.config
====================
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path] = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file. If a relative path is passed,
    attempts to find it relative to cwd first, then relative to package root.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        pkg_root = Path(__file__).resolve().parent.parent.parent
        alt_path = pkg_root / config_path
        if alt_path.exists():
            config_path = alt_path
        else:
            alt_default = pkg_root / "configs" / "default.yaml"
            if alt_default.exists():
                config_path = alt_default
            else:
                raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    return cfg


def save_config(cfg: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save dictionary configuration to YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Utility to ensure a directory exists."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
