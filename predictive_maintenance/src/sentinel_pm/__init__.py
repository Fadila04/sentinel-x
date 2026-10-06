"""Sentinel-X — maintenance prédictive par détection d'anomalies sur les capteurs ESP8266."""

from sentinel_pm.config import Config, load_config
from sentinel_pm.features import FEATURE_COLUMNS, make_features
from sentinel_pm.model import DetectorBundle, detect, load_bundle, save_bundle, train_detector
from sentinel_pm.realtime import RealTimeDetector

__version__ = "0.1.0"

__all__ = [
    "FEATURE_COLUMNS",
    "Config",
    "DetectorBundle",
    "RealTimeDetector",
    "detect",
    "load_bundle",
    "load_config",
    "make_features",
    "save_bundle",
    "train_detector",
]
