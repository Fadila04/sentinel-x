import pandas as pd
import pytest

from sentinel_pm.data import inject_event, simulate_normal


def test_simulation_is_deterministic():
    pd.testing.assert_frame_equal(simulate_normal(1, seed=0), simulate_normal(1, seed=0))


def test_temporal_split_has_no_anomaly_in_train(small_cfg, small_dataset):
    data, train, test = small_dataset
    assert len(train) + len(test) == len(data)
    assert train.index.max() < test.index.min()
    assert train["label"].sum() == 0
    expected = sum(e.duration for e in small_cfg.simulation.test_events)
    assert test["label"].sum() == expected


def test_frozen_sensor_is_constant():
    df = inject_event(simulate_normal(1, seed=0), "frozen_sensor", 100, 30)
    assert df["temperature"].iloc[100:130].nunique() == 1
    assert (df["event"].iloc[100:130] == "frozen_sensor").all()


def test_inject_event_does_not_mutate_input():
    df = simulate_normal(1, seed=0)
    before = df.copy()
    inject_event(df, "gas_spike", 10, 15)
    pd.testing.assert_frame_equal(df, before)


def test_unknown_event_raises():
    with pytest.raises(ValueError):
        inject_event(simulate_normal(1, seed=0), "meteor", 0, 10)


def test_out_of_bounds_event_raises():
    with pytest.raises(IndexError):
        inject_event(simulate_normal(1, seed=0), "gas_spike", 1430, 30)
