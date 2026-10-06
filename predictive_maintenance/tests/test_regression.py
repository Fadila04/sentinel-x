"""Non-régression : la config par défaut doit reproduire les résultats du prototype validé
(notebooks/00_prototype_isolation_forest.ipynb, code ré-exécuté avec scikit-learn 1.8)."""

import pandas as pd
import pytest

from sentinel_pm.config import Config
from sentinel_pm.data import build_dataset
from sentinel_pm.evaluation import event_report, false_alarm_breakdown, point_metrics
from sentinel_pm.model import detect, train_detector
from sentinel_pm.realtime import RealTimeDetector

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run():
    cfg = Config()
    _, train, test = build_dataset(cfg.simulation, cfg.seed)
    bundle = train_detector(train, cfg)
    scores, alerts = detect(bundle, test)
    return cfg, bundle, test, alerts


def test_threshold(run):
    _, bundle, _, _ = run
    assert bundle.threshold == pytest.approx(-0.0670, abs=5e-4)


def test_point_metrics(run):
    _, _, test, alerts = run
    m = point_metrics(test.loc[alerts.index, "label"], alerts.astype(int))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (437, 50, 118, 2245)


def test_all_events_detected_early(run):
    cfg, _, test, alerts = run
    rep = event_report(test, alerts, cfg.evaluation.critical_temperature)
    assert rep["detected"].all()
    assert rep["detection_delay_min"].tolist() == [32, 5, 10, 8, 33]
    drifts = rep[rep["event"] == "slow_drift"]
    assert (drifts["lead_over_fixed_threshold_min"] > 120).all()


def test_false_alarms_are_mostly_tail_effect(run):
    _, bundle, test, alerts = run
    fa = false_alarm_breakdown(test.loc[alerts.index, "label"], alerts, 2 * bundle.window_long)
    assert fa == {"normal_samples": 2295, "false_alarms": 50, "tail_effect": 49, "spontaneous": 1}


def test_realtime_replay_first_alert(run):
    _, bundle, test, _ = run
    start = test.index[300]
    stream = test.loc[start - pd.Timedelta(minutes=60) : start + pd.Timedelta(minutes=240)]
    det = RealTimeDetector(bundle)
    first = next(
        p
        for ts, row in stream.iterrows()
        if (p := det.update({"timestamp": ts, **row[["temperature", "humidity", "gas"]]}))
    )
    assert first["timestamp"] == "2026-10-09T13:32:00"
    assert first["severity"] == "warning"
