"""Évaluation à deux niveaux : par mesure (P/R/F1) et par événement (délai, avance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

NORMAL_EVENT = "normal"


def point_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Chaque mesure compte comme un exemple."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def event_report(df: pd.DataFrame, alerts: pd.Series, critical_temperature: float) -> pd.DataFrame:
    """Pour chaque panne : détectée ? délai ? avance sur un seuil fixe de température ?

    Le seuil fixe sert UNIQUEMENT de point de comparaison ; il n'est pas utilisé par le modèle.
    """
    events = df.loc[alerts.index, "event"]
    run_id = (events != events.shift()).cumsum()
    is_event = events != NORMAL_EVENT

    rows = []
    for _, segment in events[is_event].groupby(run_id[is_event]):
        start, end = segment.index[0], segment.index[-1]
        window_alerts = alerts.loc[start:end]
        first_alert = window_alerts[window_alerts].index[0] if window_alerts.any() else None
        detected = first_alert is not None

        over = df.loc[start:end]
        over = over[over["temperature"] > critical_temperature]
        fixed_trigger = over.index[0] if len(over) else None

        rows.append(
            {
                "event": segment.iloc[0],
                "start": start,
                "duration_min": len(segment),
                "detected": detected,
                "detection_delay_min": _minutes(first_alert - start) if detected else np.nan,
                "lead_over_fixed_threshold_min": (
                    _minutes(fixed_trigger - first_alert)
                    if detected and fixed_trigger is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def false_alarm_breakdown(y_true: pd.Series, alerts: pd.Series, tail_window: int) -> dict[str, int]:
    """Sépare les fausses alertes « effet de queue » (juste après une panne) des spontanées.

    Juste après une panne, les fenêtres glissantes contiennent encore des mesures anormales.
    """
    normal = y_true == 0
    after_event = y_true.rolling(tail_window, min_periods=1).max().astype(bool)
    return {
        "normal_samples": int(normal.sum()),
        "false_alarms": int(alerts[normal].sum()),
        "tail_effect": int(alerts[normal & after_event].sum()),
        "spontaneous": int(alerts[normal & ~after_event].sum()),
    }


def _minutes(delta: pd.Timedelta) -> float:
    return delta.total_seconds() / 60
