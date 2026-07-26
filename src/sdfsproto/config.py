"""YAML configuration loading with deep-merge overrides and attribute access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @staticmethod
    def _wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config._wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Config._wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: Config, sets: list[str]) -> None:
    """Apply dotted-key overrides like 'model.dynamic.backend=api_seq' (YAML-parsed values)."""
    for s in sets:
        key, _, raw = s.partition("=")
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = Config._wrap(value)


def load_config(*paths: str | Path) -> Config:
    """Load one or more YAML files; later files deep-merge over earlier ones."""
    merged: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return Config._wrap(merged)
