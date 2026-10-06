"""Configuration centralisée (remplace les constantes éparpillées dans le notebook)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EventSpec:
    """Panne injectée dans la période de test (minutes relatives au début du test)."""

    kind: str
    start: int
    duration: int


@dataclass(frozen=True)
class SimulationConfig:
    start: str = "2026-10-05 08:00"
    train_days: int = 4
    test_days: int = 2
    test_events: tuple[EventSpec, ...] = (
        EventSpec("slow_drift", 300, 240),
        EventSpec("gas_spike", 900, 15),
        EventSpec("frozen_sensor", 1300, 60),
        EventSpec("overheating", 1800, 40),
        EventSpec("slow_drift", 2300, 200),
    )


@dataclass(frozen=True)
class FeatureConfig:
    window: int = 15
    window_long: int = 30

    def __post_init__(self) -> None:
        if self.window < 2 or self.window_long < 2:
            raise ValueError("Les fenêtres doivent contenir au moins 2 mesures.")


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 300
    threshold_quantile: float = 0.005
    persistence: int = 3
    critical_factor: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile doit être dans ]0, 1[.")
        if self.persistence < 1:
            raise ValueError("persistence doit être >= 1.")


@dataclass(frozen=True)
class EvaluationConfig:
    critical_temperature: float = 38.0


@dataclass(frozen=True)
class Config:
    seed: int = 42
    device_id: str = "esp8266-01"
    model_path: str = "models/sentinel_isolation_forest.joblib"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    """Charge un fichier YAML. Les clés absentes prennent les valeurs par défaut."""
    if path is None:
        return Config()

    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    sim_raw = dict(raw.pop("simulation", {}) or {})
    if "test_events" in sim_raw:
        sim_raw["test_events"] = tuple(EventSpec(**e) for e in sim_raw["test_events"])

    return Config(
        simulation=SimulationConfig(**sim_raw),
        features=FeatureConfig(**(raw.pop("features", {}) or {})),
        model=ModelConfig(**(raw.pop("model", {}) or {})),
        evaluation=EvaluationConfig(**(raw.pop("evaluation", {}) or {})),
        **raw,
    )
