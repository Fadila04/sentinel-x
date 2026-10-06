"""Graphiques (dépendance optionnelle : `pip install -e .[viz]`). Les fonctions renvoient
la figure au lieu d'appeler plt.show(), pour pouvoir l'afficher OU la sauvegarder."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_COLORS = {"temperature": "tab:orange", "humidity": "tab:blue", "gas": "tab:green"}


def shade_events(ax: Axes, df: pd.DataFrame) -> None:
    """Colorie en rouge les périodes d'anomalie réelles (vérité terrain)."""
    run_id = (df["label"] != df["label"].shift()).cumsum()
    anomalous = df["label"] == 1
    for _, g in df[anomalous].groupby(run_id[anomalous]):
        ax.axvspan(g.index[0], g.index[-1], color="red", alpha=0.15)


def plot_sensors(df: pd.DataFrame, title: str | None = None, shade: bool = True) -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, col in zip(axes, _COLORS, strict=True):
        ax.plot(df.index, df[col], color=_COLORS[col], lw=0.8)
        if shade and "label" in df:
            shade_events(ax, df)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    columns: Sequence[str] = ("temp_slope", "temp_std", "corr_temp_gas"),
) -> Figure:
    fig, axes = plt.subplots(1, len(columns), figsize=(14, 3.5))
    for ax, col in zip(axes, columns, strict=True):
        ax.hist(X_train[col], bins=60, alpha=0.7, label="train (normal)", density=True)
        faults = X_test.loc[y_test == 1, col]
        ax.hist(faults, bins=60, alpha=0.7, label="test (pannes)", density=True)
        ax.set_title(col)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_detection(
    df: pd.DataFrame, scores: pd.Series, alerts: pd.Series, threshold: float
) -> Figure:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    alert_idx = alerts[alerts].index

    for ax, col, ylabel in (
        (axes[0], "temperature", "Température (°C)"),
        (axes[1], "gas", "Gaz (MQ-2)"),
    ):
        ax.plot(df.index, df[col], lw=0.8, color=_COLORS[col])
        ax.plot(alert_idx, df.loc[alert_idx, col], "r.", ms=3, label="alerte du modèle")
        shade_events(ax, df)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left")

    axes[2].plot(scores.index, scores, lw=0.7, color="tab:purple")
    axes[2].axhline(threshold, color="red", ls="--", label="seuil appris")
    shade_events(axes[2], df)
    axes[2].set_ylabel("Score d'anomalie")
    axes[2].legend(loc="lower left")

    axes[0].set_title("Détection de l'Isolation Forest sur la période de test")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
