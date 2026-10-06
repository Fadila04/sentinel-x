"""
Évaluation du modèle Sentinel-X.

Rejoue le jeu de test mesure par mesure, exactement comme le ferait le service temps réel
(même calcul de features, même score, même règle d'alerte), puis mesure :
  - le taux de détection de chaque type d'incident ;
  - le délai de détection (combien de temps après le début de l'incident) ;
  - le nombre de fausses alertes par heure de fonctionnement normal ;
  - l'ANTICIPATION : combien de minutes avant un seuil classique le modèle donne l'alerte ;
  - précision, rappel et F1-score.

Résultats :
  - reports/evaluation_summary.csv : le tableau récapitulatif (pour le dossier)
  - figures/eval_<scenario>.png    : un graphique par scénario (pour le dossier et le PowerPoint)

Utilisation :
  python evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import features as feat
from alerting import AlertTracker, score

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "anomaly_model.joblib"
TEST_PATH = HERE / "data" / "synthetic" / "test_scenarios.csv"
FIG_DIR = HERE / "figures"
REPORT_DIR = HERE / "reports"

# "Seuils classiques" auxquels on compare le modèle (ce qu'aurait fait un simple if)
TEMP_LIMIT = 40     # °C
GAS_LIMIT = 500     # valeur brute du MQ-2

# Pour le pic de gaz (20 s), on accepte une détection jusqu'à 1 minute après la fin du pic
SPIKE_GRACE_MIN = 1.0


def minutes_since(ts, origin):
    return (ts - origin).dt.total_seconds() / 60


def replay_run(bundle, meta_run):
    """Rejoue une série mesure par mesure avec la règle d'alerte. Renvoie l'état et les débuts d'alerte."""
    tracker = AlertTracker.from_bundle(bundle)
    active, starts = [], []
    for idx, risk, fault in zip(meta_run.index, meta_run["risk"], meta_run["fault"]):
        if tracker.update(risk, fault) == "started":
            starts.append(idx)
        active.append(tracker.active)
    return np.array(active), starts


def evaluate_run(bundle, raw_run, meta_run):
    """Calcule toutes les mesures d'une série."""
    origin = raw_run["timestamp"].iloc[0]
    meta_run = meta_run.assign(t=minutes_since(meta_run["timestamp"], origin))
    active, starts = replay_run(bundle, meta_run)
    meta_run = meta_run.assign(alert=active)

    incident = raw_run[raw_run["label"] == 1]
    has_incident = not incident.empty
    t_start = minutes_since(incident["timestamp"], origin).min() if has_incident else np.inf
    t_end = minutes_since(incident["timestamp"], origin).max() if has_incident else np.inf

    # Fausses alertes : alertes qui démarrent AVANT l'incident (ou dans une série 100 % normale)
    normal_part = meta_run[meta_run["t"] < t_start]
    false_alarms = sum(1 for i in starts if meta_run.at[i, "t"] < t_start)
    normal_hours = len(normal_part) * 2 / 3600

    result = {"false_alarms": false_alarms, "normal_hours": normal_hours,
              "detected": None, "delay_min": np.nan, "threshold_min": np.nan,
              "t_start": t_start, "t_alert": np.nan, "t_threshold": np.nan}

    if has_incident:
        # Première alerte pendant l'incident (ou déjà active au moment où il commence)
        during = meta_run[meta_run["t"] >= t_start]
        limit = t_end + (SPIKE_GRACE_MIN if raw_run["scenario"].iloc[0] == "spike" else 0)
        hits = during[(during["alert"]) & (during["t"] <= limit)]
        result["detected"] = not hits.empty
        if not hits.empty:
            result["t_alert"] = hits["t"].iloc[0]
            result["delay_min"] = hits["t"].iloc[0] - t_start

        # Quand un seuil classique aurait-il réagi ?
        after = raw_run[minutes_since(raw_run["timestamp"], origin) >= t_start]
        crossed = after[(after["temp"] >= TEMP_LIMIT) | (after["gas"] >= GAS_LIMIT)]
        if not crossed.empty:
            result["t_threshold"] = minutes_since(crossed["timestamp"], origin).iloc[0]
            result["threshold_min"] = result["t_threshold"] - t_start

    return result, meta_run


def plot_run(scenario, raw_run, meta_run, res):
    """Graphique : capteurs + risque, avec début d'incident, alerte du modèle et seuil classique."""
    origin = raw_run["timestamp"].iloc[0]
    t_raw = minutes_since(raw_run["timestamp"], origin)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    axes[0].plot(t_raw, raw_run["temp"], color="tab:red")
    axes[0].axhline(TEMP_LIMIT, color="tab:red", ls=":", lw=1)
    axes[0].set_ylabel("température (°C)")
    axes[1].plot(t_raw, raw_run["gas"], color="tab:blue")
    axes[1].axhline(GAS_LIMIT, color="tab:blue", ls=":", lw=1)
    axes[1].set_ylabel("gaz (brut)")
    axes[2].plot(meta_run["t"], meta_run["risk"], color="black", lw=1)
    axes[2].axhline(50, color="gray", ls="--", lw=1)
    axes[2].fill_between(meta_run["t"], 0, 100, where=meta_run["alert"], color="green", alpha=0.15,
                         label="alerte active")
    axes[2].set_ylabel("risque (0-100)")
    axes[2].set_ylim(0, 100)

    for ax in axes:
        if np.isfinite(res["t_start"]):
            ax.axvline(res["t_start"], color="orange", lw=2)
        if np.isfinite(res["t_alert"]):
            ax.axvline(res["t_alert"], color="green", lw=2)
        if np.isfinite(res["t_threshold"]):
            ax.axvline(res["t_threshold"], color="red", lw=2, ls="--")

    legend = [plt.Line2D([], [], color="orange", lw=2, label="début de l'incident"),
              plt.Line2D([], [], color="green", lw=2, label="alerte du modèle"),
              plt.Line2D([], [], color="red", lw=2, ls="--", label="seuil classique franchi")]
    axes[0].legend(handles=legend, loc="upper left", fontsize=8)
    axes[0].set_title(f"Scénario : {scenario}", loc="left")
    axes[-1].set_xlabel("minutes")
    fig.tight_layout()
    path = FIG_DIR / f"eval_{scenario}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt_min(v):
    return "-" if pd.isna(v) else f"{v:.1f} min"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Modèle introuvable. Lance d'abord : python train.py")
    if not TEST_PATH.exists():
        raise SystemExit("Jeu de test introuvable. Lance d'abord : python synthetic.py")

    bundle = joblib.load(MODEL_PATH)
    raw = pd.read_csv(TEST_PATH, parse_dates=["timestamp"])

    # Features + score, exactement comme en temps réel
    X, meta = feat.build_features(raw, group_col="run_id")
    risk, faults = score(bundle, X)
    meta = meta.assign(risk=risk, fault=faults)

    FIG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    rows, all_meta, plotted = [], [], set()
    for run_id, raw_run in raw.groupby("run_id", sort=False):
        scenario = raw_run["scenario"].iloc[0]
        res, meta_run = evaluate_run(bundle, raw_run, meta[meta["run_id"] == run_id])
        rows.append({"scenario": scenario, "run_id": run_id, **res})
        all_meta.append(meta_run)
        if scenario not in plotted:
            plot_run(scenario, raw_run, meta_run, res)
            plotted.add(scenario)

    runs = pd.DataFrame(rows)
    all_meta = pd.concat(all_meta)

    # ---- Tableau 1 : détection par scénario ----
    incidents = runs[runs["detected"].notna()]
    summary = incidents.groupby("scenario", sort=False).agg(
        detection=("detected", "mean"),
        delai_moyen=("delay_min", "mean"),
        seuil_classique=("threshold_min", "mean"),
        seuil_jamais=("threshold_min", lambda s: s.isna().mean()),
    )
    summary["anticipation"] = summary["seuil_classique"] - summary["delai_moyen"]

    print("=" * 78)
    print("DÉTECTION DES INCIDENTS")
    print("=" * 78)
    print(f"{'scénario':<10} {'détectés':>9} {'délai modèle':>14} {'seuil classique':>17} {'anticipation':>14}")
    for name, r in summary.iterrows():
        seuil = "jamais" if r["seuil_jamais"] == 1 else fmt_min(r["seuil_classique"])
        anticip = "∞ (seuil aveugle)" if r["seuil_jamais"] == 1 else fmt_min(r["anticipation"])
        print(f"{name:<10} {r['detection']:>8.0%} {fmt_min(r['delai_moyen']):>14} {seuil:>17} {anticip:>14}")

    # ---- Tableau 2 : fausses alertes ----
    fa = runs["false_alarms"].sum()
    hours = runs["normal_hours"].sum()
    print("\n" + "=" * 78)
    print("FAUSSES ALERTES")
    print("=" * 78)
    print(f"{fa} fausse(s) alerte(s) sur {hours:.1f} h de fonctionnement normal "
          f"-> {fa / hours:.2f} par heure")

    # ---- Tableau 3 : métriques classiques (mesure par mesure) ----
    # On évalue les parties normales AVANT l'incident et les périodes d'incident.
    origin = raw.groupby("run_id")["timestamp"].transform("min")
    start = raw[raw["label"] == 1].groupby("run_id")["timestamp"].min()
    t_inc = all_meta["run_id"].map(start)
    mask = (all_meta["label"] == 1) | t_inc.isna() | (all_meta["timestamp"] < t_inc)
    p, r, f1, _ = precision_recall_fscore_support(
        all_meta.loc[mask, "label"], all_meta.loc[mask, "alert"], average="binary", zero_division=0)
    print("\n" + "=" * 78)
    print("MÉTRIQUES CLASSIQUES (mesure par mesure)")
    print("=" * 78)
    print(f"Précision : {p:.1%}  (quand le modèle alerte, il a raison dans {p:.0%} des cas)")
    print(f"Rappel    : {r:.1%}  (part des mesures d'incident pendant lesquelles l'alerte est active)")
    print(f"F1-score  : {f1:.1%}")

    # ---- Sauvegardes ----
    out = summary.copy()
    out.loc["_global", ["detection"]] = incidents["detected"].mean()
    out["fausses_alertes_par_heure"] = np.nan
    out.loc["_global", "fausses_alertes_par_heure"] = fa / hours
    out.loc["_global", ["precision", "rappel", "f1"]] = [p, r, f1]
    out.round(3).to_csv(REPORT_DIR / "evaluation_summary.csv")
    print(f"\nTableau sauvegardé : {REPORT_DIR / 'evaluation_summary.csv'}")
    print(f"Graphiques         : {FIG_DIR}/eval_<scenario>.png")


if __name__ == "__main__":
    main()