import pandas as pd
import pytest

from sentinel_pm.features import FEATURE_COLUMNS
from sentinel_pm.model import apply_persistence, detect, load_bundle, save_bundle


def test_persistence_rule():
    scores = pd.Series([0.1, -1, -1, 0.1, -1, -1, -1, -1])
    alerts = apply_persistence(scores, threshold=0, persistence=3)
    assert alerts.tolist() == [False] * 6 + [True, True]


def test_bundle_content(small_cfg, small_bundle):
    assert small_bundle.feature_columns == FEATURE_COLUMNS
    assert small_bundle.threshold < 0
    assert small_bundle.persistence == small_cfg.model.persistence
    assert "sklearn_version" in small_bundle.metadata


def test_save_load_roundtrip(tmp_path, small_bundle, small_dataset):
    _, _, test = small_dataset
    path = save_bundle(small_bundle, tmp_path / "sub" / "model.joblib")
    loaded = load_bundle(path)
    pd.testing.assert_series_equal(detect(small_bundle, test)[0], detect(loaded, test)[0])


def test_legacy_bundle_is_rejected(tmp_path):
    import joblib

    path = tmp_path / "legacy.joblib"
    joblib.dump({"model": None, "seuil": -0.1}, path)
    with pytest.raises(ValueError, match="Ré-entraîner"):
        load_bundle(path)


def test_detects_injected_faults(small_bundle, small_dataset):
    _, _, test = small_dataset
    _, alerts = detect(small_bundle, test)
    for kind in ("slow_drift", "frozen_sensor"):
        window = test.loc[alerts.index, "event"] == kind
        assert alerts[window].any(), kind
