"""Carga de config/prism.yaml con overrides y registro del valor efectivo.

Los valores por defecto viven solo en el YAML. Cada hoja es un mapa con
`value`, `unit` y `origin`. Las claves se refieren como "seccion.nombre".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "prism.yaml"
VALID_ORIGINS = {"cli", "measured", "derived"}
_LEAF_FIELDS = {"value", "unit", "origin", "note"}


class ConfigError(ValueError):
    pass


@dataclass
class Param:
    value: Any
    unit: str
    origin: str
    source: str = "default"
    note: str = ""


@dataclass
class PrismConfig:
    params: Dict[str, Param] = field(default_factory=dict)
    path: Optional[str] = None

    def get(self, key: str) -> Any:
        try:
            return self.params[key].value
        except KeyError:
            raise ConfigError(f"unknown config key: {key}") from None

    def to_effective_dict(self) -> Dict[str, Any]:
        return {
            "config_file": self.path,
            "params": {
                k: {"value": p.value, "unit": p.unit, "origin": p.origin, "source": p.source}
                for k, p in sorted(self.params.items())
            },
        }

    def write_effective(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        target = out / "effective_config.json"
        target.write_text(json.dumps(self.to_effective_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target


def _flatten(raw: Mapping[str, Any]) -> Dict[str, Param]:
    params: Dict[str, Param] = {}
    for section, entries in raw.items():
        if not isinstance(entries, Mapping):
            raise ConfigError(f"section '{section}' must be a mapping")
        for name, leaf in entries.items():
            key = f"{section}.{name}"
            if not isinstance(leaf, Mapping) or "value" not in leaf:
                raise ConfigError(f"{key}: expected a mapping with 'value'")
            extra = set(leaf) - _LEAF_FIELDS
            if extra:
                raise ConfigError(f"{key}: unexpected fields {sorted(extra)}")
            unit, origin = leaf.get("unit"), leaf.get("origin")
            if not unit:
                raise ConfigError(f"{key}: missing 'unit'")
            if origin not in VALID_ORIGINS:
                raise ConfigError(f"{key}: origin must be one of {sorted(VALID_ORIGINS)}, got {origin!r}")
            params[key] = Param(leaf["value"], str(unit), origin, note=str(leaf.get("note", "")))
    return params


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> PrismConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    params = _flatten(raw)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in params:
            raise ConfigError(f"unknown config key in overrides: {key}")
        params[key].value = value
        params[key].source = "override"
    return PrismConfig(params=params, path=str(cfg_path))
