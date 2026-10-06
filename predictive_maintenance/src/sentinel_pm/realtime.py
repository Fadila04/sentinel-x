"""Détecteur temps réel : consomme les mesures une par une et émet un payload d'alerte."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, TypedDict

import pandas as pd

from sentinel_pm.features import SENSOR_COLUMNS, make_features, min_history
from sentinel_pm.model import DetectorBundle


class AlertPayload(TypedDict):
    """Contrat envoyé à POST /api/v1/alerts (à valider avec l'équipe DEV)."""

    device_id: str
    type: str
    severity: str
    score: float
    timestamp: str
    measures: dict[str, float]


class RealTimeDetector:
    """Garde juste assez d'historique pour calculer les features de la mesure courante.

    Garantit les mêmes décisions que `model.detect` en batch (testé dans tests/test_realtime.py).
    """

    ALERT_TYPE = "predictive_anomaly"

    def __init__(self, bundle: DetectorBundle, device_id: str = "esp8266-01") -> None:
        self.bundle = bundle
        self.device_id = device_id
        self._buffer: deque[Mapping[str, Any]] = deque(
            maxlen=min_history(bundle.window, bundle.window_long)
        )
        self._recent: deque[bool] = deque(maxlen=bundle.persistence)

    @property
    def ready(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen

    def reset(self) -> None:
        self._buffer.clear()
        self._recent.clear()

    def score(self) -> float | None:
        """Score de la dernière mesure reçue (None si historique insuffisant)."""
        if not self.ready:
            return None
        window = pd.DataFrame(list(self._buffer)).set_index("timestamp")
        feats = make_features(window, self.bundle.window, self.bundle.window_long)
        last = feats[self.bundle.feature_columns].iloc[[-1]]
        return float(self.bundle.model.decision_function(last)[0])

    def update(self, measurement: Mapping[str, Any]) -> AlertPayload | None:
        """measurement = {"timestamp", "temperature", "humidity", "gas", ...}."""
        self._buffer.append(measurement)
        score = self.score()
        if score is None:
            return None

        self._recent.append(score < self.bundle.threshold)
        if len(self._recent) == self.bundle.persistence and all(self._recent):
            return self._build_payload(measurement, score)
        return None

    def _build_payload(self, measurement: Mapping[str, Any], score: float) -> AlertPayload:
        threshold = self.bundle.threshold
        critical = score < threshold * self.bundle.critical_factor
        return AlertPayload(
            device_id=self.device_id,
            type=self.ALERT_TYPE,
            severity="critical" if critical else "warning",
            score=round(score, 4),
            timestamp=pd.Timestamp(measurement["timestamp"]).isoformat(),
            measures={k: round(float(measurement[k]), 2) for k in SENSOR_COLUMNS},
        )
