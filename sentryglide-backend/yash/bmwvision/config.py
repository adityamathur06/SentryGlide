"""Configuration loading. The YAML file is the single source of tunable values."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["_base_dir"] = str(path.resolve().parent)
    # YAML may parse marker ids as ints or strings; normalise to int.
    m = cfg["geometry"]["markers"]
    m["positions_mm"] = {int(k): tuple(v) for k, v in m["positions_mm"].items()}
    return cfg


def resolve(cfg: dict, rel: str) -> Path:
    """Resolve a path from the config relative to the config file's folder."""
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_base_dir"]) / p
