import numpy as np
import pandas as pd
import pytest

from sentinel_pm.data import inject_event, simulate_normal
from sentinel_pm.features import FEATURE_COLUMNS, make_features, min_history


def test_columns_and_no_nan():
    f = make_features(simulate_normal(1, seed=0), 15, 30)
    assert list(f.columns) == FEATURE_COLUMNS
    assert not f.isna().any().any()


def test_min_history_yields_exactly_one_row():
    df = simulate_normal(1, seed=0)
    n = min_history(15, 30)
    assert len(make_features(df.iloc[:n], 15, 30)) == 1
    assert len(make_features(df.iloc[: n - 1], 15, 30)) == 0


def test_frozen_sensor_gives_extreme_log_std():
    df = inject_event(simulate_normal(1, seed=0), "frozen_sensor", 200, 60)
    f = make_features(df, 15, 30)
    frozen = f.loc[df.index[230]]
    assert frozen["temp_std"] == np.log10(1e-3)  # exactement, sans résidu numérique
    assert frozen["corr_temp_gas"] == 0  # corrélation indéfinie => 0, jamais ±inf


def test_features_are_finite_and_corr_bounded():
    df = inject_event(simulate_normal(1, seed=0), "frozen_sensor", 200, 120)
    f = make_features(df, 15, 30)
    assert np.isfinite(f.to_numpy()).all()
    assert f["corr_temp_gas"].between(-1, 1).all()


def test_batch_equals_isolated_window():
    """Les features d'une ligne ne dépendent pas de l'historique au-delà de la fenêtre."""
    df = inject_event(simulate_normal(1, seed=0), "frozen_sensor", 200, 120)
    batch = make_features(df, 15, 30)
    n = min_history(15, 30)
    for i in (150, 240, 300):
        iso = make_features(df.iloc[i - n + 1 : i + 1], 15, 30)
        pd.testing.assert_series_equal(
            iso.iloc[-1], batch.loc[df.index[i]], check_names=False, atol=1e-9
        )


def test_missing_sensor_column_raises():
    with pytest.raises(KeyError):
        make_features(pd.DataFrame({"temperature": [1.0]}), 15, 30)
