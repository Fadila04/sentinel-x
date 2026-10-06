"""Entraînement, seuil appris, règle de persistance et persistance disque du détecteur."""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from sentinel_pm.config import Config
from sentinel_pm.features import FEATURE_COLUMNS, make_features

logger = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION = 1


@dataclass
class DetectorBundle:
    """Tout ce qu'il faut pour rejouer la détection à l'identique en production."""

    model: IsolationForest
    threshold: float
    feature_columns: list[str]
    window: int
    window_long: int
    persistence: int
    critical_factor: float = 2.0
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Briques élémentaires -------------------------------------------------------------


def fit_isolation_forest(X: pd.DataFrame, n_estimators: int, seed: int) -> IsolationForest:
    """Entraîne l'Isolation Forest sur des données NORMALES uniquement."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",  # le seuil est choisi par nous (learn_threshold)
        random_state=seed,
        n_jobs=-1,
    )
    return model.fit(X)


def anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> pd.Series:
    """Score de décision : plus il est négatif, plus le point est anormal."""
    return pd.Series(model.decision_function(X), index=X.index, name="score")


def learn_threshold(model: IsolationForest, X_normal: pd.DataFrame, quantile: float) -> float:
    """Seuil = percentile bas des scores observés en fonctionnement normal."""
    return float(np.quantile(model.decision_function(X_normal), quantile))


def apply_persistence(scores: pd.Series, threshold: float, persistence: int) -> pd.Series:
    """Alerte si `persistence` mesures consécutives sont sous le seuil."""
    below = (scores < threshold).astype(int)
    return (below.rolling(persistence).sum() == persistence).rename("alert")


# --- Pipeline -------------------------------------------------------------------------


def train_detector(train_df: pd.DataFrame, cfg: Config) -> DetectorBundle:
    """Mesures brutes normales -> détecteur prêt à l'emploi (modèle + seuil + config)."""
    fc, mc = cfg.features, cfg.model
    X_train = make_features(train_df, fc.window, fc.window_long)
    logger.info("Entraînement sur %d échantillons, %d features", *X_train.shape)

    model = fit_isolation_forest(X_train, mc.n_estimators, cfg.seed)
    threshold = learn_threshold(model, X_train, mc.threshold_quantile)
    logger.info("Seuil de score appris : %.4f", threshold)

    return DetectorBundle(
        model=model,
        threshold=threshold,
        feature_columns=list(X_train.columns),
        window=fc.window,
        window_long=fc.window_long,
        persistence=mc.persistence,
        critical_factor=mc.critical_factor,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_train_samples": int(len(X_train)),
            "train_period": [str(train_df.index[0]), str(train_df.index[-1])],
            "sklearn_version": sklearn.__version__,
            "config": cfg.to_dict(),
        },
    )


def detect(bundle: DetectorBundle, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Détection batch sur des mesures brutes. Retourne (scores, alertes booléennes)."""
    X = make_features(df, bundle.window, bundle.window_long)[bundle.feature_columns]
    scores = anomaly_scores(bundle.model, X)
    return scores, apply_persistence(scores, bundle.threshold, bundle.persistence)


# --- Sérialisation --------------------------------------------------------------------


def save_bundle(bundle: DetectorBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": BUNDLE_FORMAT_VERSION, **asdict(bundle)}
    payload["model"] = bundle.model  # asdict copie profondément ; on garde l'objet tel quel
    joblib.dump(payload, path)
    logger.info("Détecteur sauvegardé : %s", path)
    return path


def load_bundle(path: str | Path) -> DetectorBundle:
    payload = joblib.load(Path(path))
    version = payload.pop("format_version", None)
    if version != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Format de bundle non supporté ({version!r}). Ré-entraîner avec `sentinel-pm train`."
        )

    trained_with = payload.get("metadata", {}).get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Modèle entraîné avec scikit-learn {trained_with}, "
            f"chargé avec {sklearn.__version__} : résultats possiblement différents.",
            stacklevel=2,
        )

    bundle = DetectorBundle(**payload)
    if bundle.feature_columns != FEATURE_COLUMNS:
        raise ValueError("Les features du bundle ne correspondent pas à la version du code.")
    return bundle
