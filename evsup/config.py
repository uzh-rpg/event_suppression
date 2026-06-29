from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def merge_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config)
    for dotted_key, value in overrides.items():
        cursor = merged
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return merged


def get_device(gpu: int | None = None) -> torch.device:
    if torch.cuda.is_available():
        if gpu is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")
