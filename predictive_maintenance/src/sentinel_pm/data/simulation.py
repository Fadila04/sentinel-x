"""Simulateur de mesures ESP8266 (DHT22, MQ-2, PIR) et injection de pannes.

Fréquence : 1 mesure / minute. `label` (0 normal / 1 anomalie) et `event` servent
UNIQUEMENT à l'évaluation : le modèle n'en a jamais besoin pour s'entraîner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel_pm.config import SimulationConfig

MINUTES_PER_DAY = 24 * 60

SLOW_DRIFT = "slow_drift"  # T° et gaz montent lentement ET ensemble
GAS_SPIKE = "gas_spike"  # pic brutal de gaz puis retombée (fuite)
OVERHEATING = "overheating"  # montée rapide de la température
FROZEN_SENSOR = "frozen_sensor"  # le DHT22 renvoie toujours la même valeur

EVENT_KINDS: frozenset[str] = frozenset({SLOW_DRIFT, GAS_SPIKE, OVERHEATING, FROZEN_SENSOR})


def simulate_normal(n_days: int, seed: int, start: str = "2026-10-05 08:00") -> pd.DataFrame:
    """Simule le fonctionnement NORMAL des capteurs."""
    rng = np.random.default_rng(seed)
    n = n_days * MINUTES_PER_DAY
    idx = pd.date_range(start, periods=n, freq="1min", name="timestamp")
    hours = idx.hour + idx.minute / 60

    # Température : 22 °C ± 4 °C selon l'heure, bruit lissé (inertie du capteur)
    temp = 22 + 4 * np.sin(2 * np.pi * (hours - 9) / 24)
    temp = temp + pd.Series(rng.normal(0, 0.35, n)).ewm(span=3).mean().to_numpy()

    # Humidité : anti-corrélée à la température
    hum = 55 - 1.2 * (temp - 22) + rng.normal(0, 1.5, n)

    # Gaz MQ-2 : valeur brute ADC (0-1023), repos ~180, légère variation journalière
    gas = 180 + 8 * np.sin(2 * np.pi * (hours - 14) / 24) + rng.normal(0, 5, n)

    # Présence PIR : déclenchements rares (non utilisé par ce modèle)
    pir = (rng.random(n) < 0.02).astype(int)

    df = pd.DataFrame({"temperature": temp, "humidity": hum, "gas": gas, "pir": pir}, index=idx)
    df["label"] = 0
    df["event"] = "normal"
    return df


def inject_event(df: pd.DataFrame, kind: str, start_min: int, dur_min: int) -> pd.DataFrame:
    """Retourne une copie de `df` avec une panne injectée à la ligne `start_min`."""
    if kind not in EVENT_KINDS:
        raise ValueError(f"Type de panne inconnu : {kind!r} (attendu : {sorted(EVENT_KINDS)})")
    if start_min < 0 or start_min + dur_min > len(df):
        raise IndexError("La panne dépasse les bornes de la série.")

    df = df.copy()
    s, e = start_min, start_min + dur_min
    k = np.arange(dur_min)
    r = k / dur_min  # progression 0 -> 1
    temp, hum, gas = (df[c].to_numpy(copy=True) for c in ("temperature", "humidity", "gas"))

    if kind == SLOW_DRIFT:
        d_temp = 20 * r**1.5  # démarre doucement, accélère
        temp[s:e] += d_temp
        gas[s:e] += 160 * r**1.5
        hum[s:e] -= 0.5 * d_temp
    elif kind == GAS_SPIKE:
        profile = np.minimum(k / 3, 1) * np.exp(-k / (dur_min / 2))
        gas[s:e] += 260 * profile
    elif kind == OVERHEATING:
        d_temp = 12 * np.minimum(k / (dur_min * 0.5), 1)
        temp[s:e] += d_temp
        hum[s:e] -= 0.8 * d_temp
    elif kind == FROZEN_SENSOR:
        temp[s:e] = temp[s]
        hum[s:e] = hum[s]

    df["temperature"], df["humidity"], df["gas"] = temp, hum, np.clip(gas, 0, 1023)
    df.iloc[s:e, df.columns.get_loc("label")] = 1
    df.iloc[s:e, df.columns.get_loc("event")] = kind
    return df


def build_dataset(
    cfg: SimulationConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construit (data, train, test) avec une découpe TEMPORELLE (jamais de shuffle).

    train = calibration sans panne ; test = période suivante avec les pannes de `cfg`.
    """
    data = simulate_normal(cfg.train_days + cfg.test_days, seed=seed, start=cfg.start)
    t0 = cfg.train_days * MINUTES_PER_DAY
    for ev in cfg.test_events:
        data = inject_event(data, ev.kind, t0 + ev.start, ev.duration)
    return data, data.iloc[:t0], data.iloc[t0:]
