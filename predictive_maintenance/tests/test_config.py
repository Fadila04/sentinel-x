from pathlib import Path

import pytest

from sentinel_pm.config import Config, ModelConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_builtin_defaults():
    assert load_config(ROOT / "configs" / "default.yaml") == Config()


def test_partial_yaml_keeps_defaults(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  persistence: 5\n")
    cfg = load_config(p)
    assert cfg.model.persistence == 5
    assert cfg.model.n_estimators == Config().model.n_estimators


@pytest.mark.parametrize("kwargs", [{"threshold_quantile": 0}, {"persistence": 0}])
def test_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
