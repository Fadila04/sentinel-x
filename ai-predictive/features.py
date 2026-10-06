"""
Calcul des features Sentinel-X.

Transforme les mesures brutes (une ligne toutes les 2 s) en "résumés de la dernière minute"
que le modèle d'IA peut comprendre : niveau moyen, agitation, vitesse de montée, dérive,
et le fait que la température et le gaz bougent ensemble.

IMPORTANT : ce fichier est utilisé À LA FOIS par l'entraînement (train.py) et par le
service temps réel (service.py). Ne jamais recopier ces calculs ailleurs : si les deux
calculent les features différemment, le modèle donne n'importe quoi en production.

Pour vérifier que les features fonctionnent :
  python features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Réglages
# ---------------------------------------------------------------------------
SENSORS = ["temp", "hum", "gas"]

# Valeurs physiquement possibles. En dehors : c'est un faux contact, on jette la mesure.
LIMITS = {"temp": (-20, 80), "hum": (0, 100), "gas": (0, 1023)}

WINDOW = 30          # fenêtre courte : 30 mesures = 1 minute
HALF = WINDOW // 2   # demi-fenêtre, pour calculer la pente
LONG_WINDOW = 150    # fenêtre longue : 150 mesures = 5 minutes ("l'habitude récente")

# Nombre de mesures nécessaires avant de pouvoir calculer la première feature.
# En temps réel, le service devra attendre ce nombre de mesures (5 min) avant de prédire.
REQUIRED_HISTORY = LONG_WINDOW
BUFFER_SIZE = LONG_WINDOW + 20   # petite marge pour absorber les lectures ratées

FEATURE_COLUMNS = (
    [f"{c}_{kind}" for c in SENSORS for kind in ["mean", "std", "slope", "drift"]]
    + ["corr_temp_gas", "slope_product"]
)


# ---------------------------------------------------------------------------
# 1. Nettoyage
# ---------------------------------------------------------------------------
def clean(df):
    """Remet les données en état : valeurs impossibles supprimées, trous bouchés."""
    df = df.copy()
    for col in SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")      # texte bizarre -> vide
        low, high = LIMITS[col]
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
    # Une lecture ratée est remplacée par la précédente (5 trous d'affilée maximum)
    df[SENSORS] = df[SENSORS].ffill(limit=5)
    return df.dropna(subset=SENSORS)


# ---------------------------------------------------------------------------
# 2. Features sur une série continue
# ---------------------------------------------------------------------------
def _features_one_series(df):
    """Calcule les features pour UNE série continue (une session, une série de test...)."""
    f = pd.DataFrame(index=df.index)

    for col in SENSORS:
        s = df[col]
        # Niveau moyen sur la dernière minute
        f[f"{col}_mean"] = s.rolling(WINDOW).mean()
        # Agitation sur la dernière minute, en logarithme : un capteur figé (agitation = 0)
        # devient alors TRÈS différent d'un capteur normal, ce qui aide le modèle à le repérer
        f[f"{col}_std"] = np.log10(s.rolling(WINDOW).std() + 1e-3)
        # Pente : moyenne des 30 dernières secondes moins moyenne des 30 secondes d'avant
        half_mean = s.rolling(HALF).mean()
        f[f"{col}_slope"] = (half_mean - half_mean.shift(HALF)) / HALF
        # Dérive : niveau de la dernière minute comparé aux 5 dernières minutes
        f[f"{col}_drift"] = f[f"{col}_mean"] - s.rolling(LONG_WINDOW).mean()

    # Est-ce que température et gaz bougent ensemble ? (entre -1 et +1)
    corr = df["temp"].rolling(WINDOW).corr(df["gas"])
    f["corr_temp_gas"] = corr.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Fort uniquement quand les DEUX pentes sont fortes en même temps
    f["slope_product"] = f["temp_slope"] * f["gas_slope"]

    return f.dropna()


# ---------------------------------------------------------------------------
# 3. Fonction principale pour l'entraînement et l'évaluation
# ---------------------------------------------------------------------------
def build_features(df, group_col=None, warmup=0):
    """
    Nettoie les données puis calcule les features.

    group_col : colonne qui sépare les séries indépendantes (ex: "run_id").
                Évite qu'une fenêtre "à cheval" sur deux séries mélange leurs valeurs.
    warmup    : nombre de mesures à ignorer au début de chaque série
                (ex: 90 = 3 minutes, le temps que le MQ-2 chauffe).

    Renvoie (X, meta) :
      X    : le tableau des features, colonnes dans l'ordre de FEATURE_COLUMNS
      meta : les lignes d'origine correspondantes (timestamp, label, scenario...)
    """
    df = clean(df)
    series = [g for _, g in df.groupby(group_col, sort=False)] if group_col else [df]
    parts = [_features_one_series(s.iloc[warmup:]) for s in series]
    X = pd.concat(parts)[FEATURE_COLUMNS]
    return X, df.loc[X.index]


# ---------------------------------------------------------------------------
# 4. Diagnostic de santé des capteurs (complément du modèle d'IA)
# ---------------------------------------------------------------------------
# Un capteur figé est une PANNE, pas une anomalie de l'environnement.
# Isolation Forest la repère mal (il juge mal les valeurs plus extrêmes que tout ce qu'il a vu),
# donc on la vérifie à part : un vrai capteur bouge toujours un peu.
STUCK_LOG_STD = -2.5    # agitation < ~0.002 sur 1 minute = capteur figé


def sensor_faults(X):
    """
    Renvoie, pour chaque ligne de features, si un capteur semble figé.
      dht22_stuck : température ET humidité figées (le DHT22 mesure les deux)
      mq2_stuck   : gaz figé (un MQ-2 qui fonctionne a toujours un peu de bruit)
    """
    return pd.DataFrame({
        "dht22_stuck": (X["temp_std"] < STUCK_LOG_STD) & (X["hum_std"] < STUCK_LOG_STD),
        "mq2_stuck": X["gas_std"] < STUCK_LOG_STD,
    }, index=X.index)


# ---------------------------------------------------------------------------
# 5. Fonction pour le temps réel
# ---------------------------------------------------------------------------
def features_from_history(rows):
    """
    Pour le service temps réel : reçoit les dernières mesures (liste de dictionnaires)
    et renvoie les features de la mesure la plus récente, ou None s'il n'y a pas assez d'historique.
    """
    if len(rows) < REQUIRED_HISTORY:
        return None
    X, _ = build_features(pd.DataFrame(list(rows)))
    if X.empty:
        return None
    return X.iloc[[-1]]


# ---------------------------------------------------------------------------
# Vérification : lancer "python features.py"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    here = Path(__file__).parent
    test_path = here / "data" / "synthetic" / "test_scenarios.csv"
    if not test_path.exists():
        raise SystemExit("Lance d'abord : python synthetic.py")

    test = pd.read_csv(test_path)
    X, meta = build_features(test, group_col="run_id")
    print(f"{len(test)} mesures brutes -> {len(X)} lignes de features, {X.shape[1]} features chacune\n")

    # Moyenne de quelques features : partie normale vs pendant l'incident, par scénario
    shown = ["temp_slope", "gas_slope", "temp_drift", "gas_drift", "temp_std", "corr_temp_gas"]
    phase = np.where(meta["label"] == 1, "incident", "normal")
    table = X[shown].groupby([meta["scenario"], phase]).mean()
    pd.set_option("display.width", 140)
    print("Valeur moyenne des features (normal vs incident) :")
    print(table.round(3).to_string())

    # Graphique des features sur une série "combined"
    run = meta[meta["scenario"] == "combined"]["run_id"].min()
    sel = meta["run_id"] == run
    minutes = np.arange(sel.sum()) * 2 / 60
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    plots = [("temp", "température brute (°C)"), ("temp_slope", "pente température"),
             ("gas_slope", "pente gaz"), ("corr_temp_gas", "corrélation température/gaz")]
    for ax, (col, title) in zip(axes, plots):
        values = meta.loc[sel, col] if col == "temp" else X.loc[sel, col]
        ax.plot(minutes, values.values)
        ax.axvspan(minutes[meta.loc[sel, "label"].values == 1].min(), minutes[-1], color="orange", alpha=0.15)
        ax.set_title(title, loc="left", fontsize=10)
    axes[-1].set_xlabel("minutes (zone orange = incident)")
    fig.tight_layout()
    out = here / "figures" / "features_combined.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nGraphique : {out}")