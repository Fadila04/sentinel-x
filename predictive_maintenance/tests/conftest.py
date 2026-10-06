import pytest

from sentinel_pm.config import Config, EventSpec, ModelConfig, SimulationConfig
from sentinel_pm.data import build_dataset
from sentinel_pm.model import train_detector


@pytest.fixture(scope="session")
def small_cfg() -> Config:
    """Config réduite pour des tests rapides (1 jour de calibration, 1 jour de test)."""
    return Config(
        simulation=SimulationConfig(
            train_days=1,
            test_days=1,
            test_events=(EventSpec("slow_drift", 200, 180), EventSpec("frozen_sensor", 900, 60)),
        ),
        model=ModelConfig(n_estimators=50),
    )


@pytest.fixture(scope="session")
def small_dataset(small_cfg):
    return build_dataset(small_cfg.simulation, small_cfg.seed)


@pytest.fixture(scope="session")
def small_bundle(small_cfg, small_dataset):
    _, train, _ = small_dataset
    return train_detector(train, small_cfg)
