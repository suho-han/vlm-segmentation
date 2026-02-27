"""
I/O helpers: YAML, JSON, logging.
"""

import json
import logging
import sys
from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_json_serialisable)


def _json_serialisable(obj):
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def get_logger(name: str = "vlm-seg", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
