"""Feature engineering : transforme les mesures brutes en variables « cinétiques ».

Ce module est la SEULE source de vérité des features : il est utilisé à la fois à
l'entraînement (batch) et en production (temps réel). Ne jamais dupliquer ce calcul.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLUMNS: tuple[str, ...] = ("temperature", "humidity", "gas")

FEATURE_COLUMNS: list[str] = [
    "temperature",
    "humidity",
    "gas",
    "temp_slope",
    "gas_slope",
    "temp_std",
    "hum_std",
    "gas_std",
    "temp_amp",
    "hum_amp",
    "gas_amp",
    "corr_temp_gas",
]

_LOG_EPS = 1e-3  # évite log10(0) quand un capteur est figé

# En dessous de cet écart-type, un signal est considéré comme strictement plat.
# Nécessaire car les fenêtres glissantes de pandas sont calculées de façon incrémentale :
# sur un capteur figé, elles laissent un résidu (~1e-6) au lieu de 0, et la corrélation
# devient du bruit numérique (voire ±inf). Sans ce seuil, batch et temps réel divergent.
_FLAT_STD = 1e-4


def _log10(x: pd.Series) -> pd.Series:
    # Échelle log : un capteur figé (variabilité = 0) devient une valeur TRÈS éloignée du normal.
    return np.log10(x + _LOG_EPS)


def _rolling_std(x: pd.Series, window: int) -> pd.Series:
    std = x.rolling(window).std()
    return std.mask(std < _FLAT_STD, 0.0)


def min_history(window: int, window_long: int) -> int:
    """Nombre minimal de mesures pour que la dernière ligne ait toutes ses features."""
    return max(window, window_long + 1)


def make_features(df: pd.DataFrame, window: int, window_long: int) -> pd.DataFrame:
    """Calcule les features sur une série de mesures indexée par le temps.

    | Feature                      | Ce qu'elle détecte                         |
    |------------------------------|--------------------------------------------|
    | valeurs brutes               | valeurs hors plage                         |
    | temp_slope, gas_slope        | dérive lente (montée régulière)            |
    | *_std, *_amp                 | capteur figé (variabilité ≈ 0) ou instable |
    | corr_temp_gas                | température et gaz qui montent ensemble    |

    Les premières lignes (historique insuffisant) sont supprimées.
    """
    missing = set(SENSOR_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes capteurs manquantes : {sorted(missing)}")

    temp, hum, gas = df["temperature"], df["humidity"], df["gas"]
    f = pd.DataFrame(index=df.index)

    f["temperature"] = temp
    f["humidity"] = hum
    f["gas"] = gas

    # Pentes : variation moyenne par mesure sur la grande fenêtre
    f["temp_slope"] = (temp - temp.shift(window_long)) / window_long
    f["gas_slope"] = (gas - gas.shift(window_long)) / window_long

    # Variabilité récente
    f["temp_std"] = _log10(_rolling_std(temp, window))
    f["hum_std"] = _log10(_rolling_std(hum, window))
    f["gas_std"] = _rolling_std(gas, window)

    # Amplitude max - min : renforce la détection des signaux « plats »
    f["temp_amp"] = _log10(temp.rolling(window).max() - temp.rolling(window).min())
    f["hum_amp"] = _log10(hum.rolling(window).max() - hum.rolling(window).min())
    f["gas_amp"] = _log10(gas.rolling(window).max() - gas.rolling(window).min())

    # Corrélation glissante T°/gaz. Indéfinie si un signal est plat (capteur figé) => 0.
    flat = (_rolling_std(temp, window_long) == 0) | (_rolling_std(gas, window_long) == 0)
    corr = temp.rolling(window_long).corr(gas).mask(flat, 0.0)
    f["corr_temp_gas"] = corr.fillna(0.0).clip(-1.0, 1.0)

    return f[FEATURE_COLUMNS].dropna()
