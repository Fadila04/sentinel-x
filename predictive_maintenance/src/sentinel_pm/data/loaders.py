"""Chargement de mesures réelles (export CSV de l'ESP8266 / de la base)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentinel_pm.features import SENSOR_COLUMNS


def load_measurements_csv(path: str | Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Lit un CSV `timestamp,temperature,humidity,gas[,pir,label,event]`.

    Trie par date, supprime les doublons de timestamp et les lignes capteurs incomplètes.
    """
    df = pd.read_csv(path, parse_dates=[timestamp_col])
    missing = set(SENSOR_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"{path} : colonnes manquantes {sorted(missing)}")

    df = (
        df.dropna(subset=list(SENSOR_COLUMNS))
        .drop_duplicates(subset=timestamp_col, keep="last")
        .sort_values(timestamp_col)
        .set_index(timestamp_col)
    )
    df.index.name = "timestamp"
    return df
