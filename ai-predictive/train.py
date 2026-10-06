"""
Entraînement du modèle de détection d'anomalies Sentinel-X (Isolation Forest).

Le modèle apprend UNIQUEMENT à quoi ressemble le fonctionnement normal.
Ensuite, tout ce qui s'en écarte recevra un score de risque élevé.

Ce script sauvegarde :
  - models/anomaly_model.joblib : le modèle + tous les réglages nécessaires au service temps réel
  - models/model_info.json      : une fiche lisible (pour toi, le README et le dossier)

Exemples d'utilisation :
  python train.py                                         -> entraîne sur les données synthétiques
  python train.py --real data/raw/salle_normal_*.csv      -> ajoute de vraies mesures
  python train.py --real data/raw/salle_normal_*.csv --no-synthetic   -> uniquement les vraies
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import features as feat
from alerting import risk_from_raw

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models"
MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"
SYNTHETIC_TRAIN = HERE / "data" / "synthetic" / "train_normal.csv"
SYNTHETIC_TEST = HERE / "data" / "synthetic" / "test_scenarios.csv"

REAL_WARMUP = 90          # on ignore les 3 premières minutes de chaque vrai fichier (chauffe du MQ-2)

# Règles d'alerte utilisées par le service temps réel
ALERT_THRESHOLD = 50      # le risque doit dépasser 50...
CONSECUTIVE = 15          # ... pendant 15 mesures d'affilée (30 s) pour déclencher l'alerte
CLEAR_THRESHOLD = 40      # l'alerte se lève quand le risque redescend sous 40


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle Sentinel-X")
    parser.add_argument("--real", nargs="*", default=[],
                        help="fichiers CSV de vraies mesures NORMALES (enregistrés avec collector.py)")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="n'utilise pas les données synthétiques")
    parser.add_argument("--contamination", type=float, default=0.01,
                        help="part de mesures que le modèle considère comme suspectes à l'entraînement")
    parser.add_argument("--trees", type=int, default=200, help="nombre d'arbres de la forêt")
    parser.add_argument("--seed", type=int, default=42, help="graine du hasard (résultats reproductibles)")
    return parser.parse_args()


def load_training_features(args):
    """Charge les données normales et calcule leurs features."""
    parts, sources = [], []

    if not args.no_synthetic:
        if not SYNTHETIC_TRAIN.exists():
            raise SystemExit("Fichier synthétique introuvable. Lance d'abord : python synthetic.py")
        X, _ = feat.build_features(pd.read_csv(SYNTHETIC_TRAIN), group_col="run_id")
        parts.append(X)
        sources.append(f"synthétique ({len(X)} lignes)")

    for path in args.real:
        # Chaque fichier du collecteur = une session indépendante
        X, _ = feat.build_features(pd.read_csv(path), warmup=REAL_WARMUP)
        if X.empty:
            print(f"Attention : {path} est trop court, ignoré.")
            continue
        parts.append(X)
        sources.append(f"{Path(path).name} ({len(X)} lignes)")

    if not parts:
        raise SystemExit("Aucune donnée d'entraînement.")
    return pd.concat(parts, ignore_index=True), sources


def main():
    args = parse_args()

    # 1. Données
    X_train, sources = load_training_features(args)
    print(f"Données d'entraînement : {len(X_train)} lignes x {X_train.shape[1]} features")
    for s in sources:
        print(f"  - {s}")

    # 2. Le modèle = normalisation + Isolation Forest, rangés ensemble dans un "pipeline"
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", IsolationForest(
            n_estimators=args.trees,
            contamination=args.contamination,
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])

    # 3. Entraînement
    print("\nEntraînement en cours...")
    pipeline.fit(X_train)

    # 4. Réglage de l'échelle du score de risque
    raw = pipeline.decision_function(X_train)
    tau = float(np.std(raw))
    risk_train = risk_from_raw(raw, tau)
    print(f"Risque sur les données d'entraînement : médiane {np.median(risk_train):.0f}/100, "
          f"{(risk_train >= ALERT_THRESHOLD).mean():.1%} des mesures au-dessus de {ALERT_THRESHOLD}")

    # 5. Sauvegarde : le modèle ET tout ce qu'il faut pour l'utiliser de la même façon en temps réel
    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "feature_columns": feat.FEATURE_COLUMNS,
        "window": feat.WINDOW,
        "long_window": feat.LONG_WINDOW,
        "tau": tau,
        "alert_threshold": ALERT_THRESHOLD,
        "clear_threshold": CLEAR_THRESHOLD,
        "consecutive": CONSECUTIVE,
    }
    joblib.dump(bundle, MODEL_PATH)

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model": "IsolationForest",
        "n_estimators": args.trees,
        "contamination": args.contamination,
        "seed": args.seed,
        "training_rows": len(X_train),
        "sources": sources,
        "features": feat.FEATURE_COLUMNS,
        "window_points": feat.WINDOW,
        "long_window_points": feat.LONG_WINDOW,
        "alert_rule": f"risque >= {ALERT_THRESHOLD} pendant {CONSECUTIVE} mesures, fin sous {CLEAR_THRESHOLD}",
        "sklearn_version": sklearn.__version__,
    }
    INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModèle sauvegardé : {MODEL_PATH}")
    print(f"Fiche du modèle   : {INFO_PATH}")

    # 6. Premier contrôle rapide sur le jeu de test (l'évaluation complète sera dans evaluate.py)
    if SYNTHETIC_TEST.exists():
        test = pd.read_csv(SYNTHETIC_TEST)
        X_test, meta = feat.build_features(test, group_col="run_id")
        risk = risk_from_raw(pipeline.decision_function(X_test), tau)
        faults = feat.sensor_faults(X_test).any(axis=1)
        meta = meta.assign(risk=risk, flagged=(risk >= ALERT_THRESHOLD) | faults)
        print("\nContrôle rapide : part des mesures signalées (risque >= 50 ou capteur figé)")
        summary = meta.groupby(["scenario", "label"])["flagged"].mean().unstack()
        summary.columns = ["partie normale" if c == 0 else "pendant l'incident" for c in summary.columns]
        print((summary * 100).round(1).map(lambda v: "-" if pd.isna(v) else f"{v} %").to_string())


if __name__ == "__main__":
    main()