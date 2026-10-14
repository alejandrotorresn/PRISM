import json

import pytest

from src.core.config import ConfigError, load_config

# Valores que el codigo actual usa; el YAML debe reproducirlos.
LEGACY = {
    "solver.time_limit_s": 300,
    "solver.gap_rel": 0.01,
    "dual_placement.atol": 0.15,
    "dual_placement.rtol": 0.05,
    "runtime.pinned_cache_max_mb": 512,
    "runtime.hybrid_batch_size": 8,
    "profiler.warmup_steps": 5,
    "profiler.measure_steps": 15,
    "profiler.backward_factor_fallback": 2.0,
    "experiments.seeds": [42, 43, 44],
}


def test_defaults_reproduce_legacy_values():
    cfg = load_config()
    for key, expected in LEGACY.items():
        assert cfg.get(key) == expected, key


def test_every_param_has_unit_and_valid_origin():
    for key, p in load_config().params.items():
        assert p.unit and p.origin in {"cli", "measured", "derived"}, key


def test_override_is_recorded_and_unknown_key_rejected(tmp_path):
    cfg = load_config(overrides={"solver.time_limit_s": 60, "solver.gap_rel": None})
    assert cfg.get("solver.time_limit_s") == 60
    assert cfg.params["solver.time_limit_s"].source == "override"
    assert cfg.params["solver.gap_rel"].source == "default"
    with pytest.raises(ConfigError):
        load_config(overrides={"solver.nope": 1})
    with pytest.raises(ConfigError):
        cfg.get("solver.nope")


def test_effective_config_is_written(tmp_path):
    cfg = load_config(overrides={"runtime.optimizer": "AdamW"})
    target = cfg.write_effective(tmp_path / "run")
    data = json.loads(target.read_text())
    assert data["params"]["runtime.optimizer"]["value"] == "AdamW"
    assert data["params"]["runtime.optimizer"]["source"] == "override"
    assert data["params"]["solver.gap_rel"]["unit"] == "ratio"


def test_malformed_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("solver:\n  x:\n    value: 1\n    origin: cli\n")
    with pytest.raises(ConfigError):
        load_config(bad)
