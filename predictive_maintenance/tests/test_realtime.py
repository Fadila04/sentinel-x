import pandas as pd

from sentinel_pm.features import min_history
from sentinel_pm.model import detect
from sentinel_pm.realtime import RealTimeDetector


def _stream(detector, df):
    out = {}
    for ts, row in df.iterrows():
        out[ts] = detector.update({"timestamp": ts, **row[["temperature", "humidity", "gas"]]})
    return out


def test_no_output_before_enough_history(small_bundle, small_dataset):
    _, _, test = small_dataset
    det = RealTimeDetector(small_bundle)
    n = min_history(small_bundle.window, small_bundle.window_long)
    _stream(det, test.iloc[: n - 1])
    assert not det.ready and det.score() is None


def test_realtime_matches_batch(small_bundle, small_dataset):
    """Garantie clé : le flux temps réel prend EXACTEMENT les mêmes décisions que le batch."""
    _, _, test = small_dataset
    _, batch_alerts = detect(small_bundle, test)
    online = _stream(RealTimeDetector(small_bundle), test)
    online_alerts = pd.Series({ts: p is not None for ts, p in online.items()})
    pd.testing.assert_series_equal(
        online_alerts.loc[batch_alerts.index], batch_alerts, check_names=False, check_freq=False
    )


def test_payload_format(small_bundle, small_dataset):
    _, _, test = small_dataset
    payloads = [p for p in _stream(RealTimeDetector(small_bundle, "dev-42"), test).values() if p]
    assert payloads
    p = payloads[0]
    assert set(p) == {"device_id", "type", "severity", "score", "timestamp", "measures"}
    assert p["device_id"] == "dev-42" and p["type"] == "predictive_anomaly"
    assert p["severity"] in {"warning", "critical"}
    assert set(p["measures"]) == {"temperature", "humidity", "gas"}
    pd.Timestamp(p["timestamp"])  # ISO 8601 parsable
