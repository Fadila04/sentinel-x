"""Interface en ligne de commande.

sentinel-pm train    [--config CFG] [--csv mesures.csv]
sentinel-pm evaluate [--config CFG] [--figures DIR] [--report report.json]
sentinel-pm replay   [--config CFG] [--event-index 0]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from sentinel_pm.config import Config, load_config
from sentinel_pm.data import build_dataset, load_measurements_csv
from sentinel_pm.evaluation import event_report, false_alarm_breakdown, point_metrics
from sentinel_pm.model import detect, load_bundle, save_bundle, train_detector
from sentinel_pm.realtime import RealTimeDetector

logger = logging.getLogger("sentinel_pm")


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    if args.csv:
        train = load_measurements_csv(args.csv)
        logger.info("Données réelles : %s (%d mesures)", args.csv, len(train))
    else:
        _, train, _ = build_dataset(cfg.simulation, cfg.seed)
        logger.info("Données simulées : %d mesures de calibration", len(train))

    bundle = train_detector(train, cfg)
    path = save_bundle(bundle, args.output or cfg.model_path)
    print(f"Seuil appris : {bundle.threshold:.4f}")
    print(f"Détecteur sauvegardé : {path}")
    return 0


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> int:
    bundle = load_bundle(args.model or cfg.model_path)
    _, train, test = build_dataset(cfg.simulation, cfg.seed)

    scores, alerts = detect(bundle, test)
    y_true = test.loc[alerts.index, "label"]

    metrics = point_metrics(y_true, alerts.astype(int))
    events = event_report(test, alerts, cfg.evaluation.critical_temperature)
    false_alarms = false_alarm_breakdown(y_true, alerts, tail_window=2 * bundle.window_long)

    print("\n== Métriques par mesure ==")
    print(pd.Series(metrics).round(3).to_string())
    print("\n== Rapport par événement ==")
    print(events.to_string(index=False))
    print(f"\nÉvénements détectés : {int(events['detected'].sum())}/{len(events)}")
    print(
        f"Fausses alertes : {false_alarms['false_alarms']} / {false_alarms['normal_samples']} "
        f"(effet de queue : {false_alarms['tail_effect']}, "
        f"spontanées : {false_alarms['spontaneous']})"
    )

    if args.report:
        report = {
            "point_metrics": metrics,
            "events": json.loads(events.to_json(orient="records", date_format="iso")),
            "false_alarms": false_alarms,
            "threshold": bundle.threshold,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Rapport JSON : {args.report}")

    if args.figures:
        _save_figures(Path(args.figures), bundle, train, test, scores, alerts, y_true)
    return 0


def cmd_replay(cfg: Config, args: argparse.Namespace) -> int:
    """Rejoue une panne mesure par mesure, comme en production."""
    bundle = load_bundle(args.model or cfg.model_path)
    _, _, test = build_dataset(cfg.simulation, cfg.seed)

    event = cfg.simulation.test_events[args.event_index]
    event_start = test.index[event.start]
    stream = test.loc[
        event_start - pd.Timedelta(minutes=args.warmup) : event_start
        + pd.Timedelta(minutes=event.duration)
    ]

    detector = RealTimeDetector(bundle, device_id=cfg.device_id)
    for ts, row in stream.iterrows():
        alert = detector.update({"timestamp": ts, **row[["temperature", "humidity", "gas"]]})
        if alert:
            print(f"Panne rejouée : {event.kind}, début réel {event_start}")
            print("Payload qui serait envoyé à POST /api/v1/alerts :")
            print(json.dumps(alert, indent=2, ensure_ascii=False))
            return 0

    print(f"Aucune alerte pendant la panne {event.kind} rejouée.")
    return 1


def _save_figures(out_dir, bundle, train, test, scores, alerts, y_true) -> None:
    try:
        from sentinel_pm import visualization as viz
        from sentinel_pm.features import make_features
    except ImportError:  # matplotlib absent
        logger.warning("matplotlib non installé : `pip install -e .[viz]`")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    X_train = make_features(train, bundle.window, bundle.window_long)
    X_test = make_features(test, bundle.window, bundle.window_long)
    figures = {
        "test_period.png": viz.plot_sensors(test, "Période de test — zones rouges = pannes"),
        "feature_distributions.png": viz.plot_feature_distributions(X_train, X_test, y_true),
        "detection.png": viz.plot_detection(test, scores, alerts, bundle.threshold),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / name, dpi=120)
    print(f"Figures : {out_dir}/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel-pm", description=__doc__.split("\n")[0])
    parser.add_argument("-c", "--config", default=None, help="Fichier YAML (défaut : intégré)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Entraîne et sauvegarde le détecteur")
    p.add_argument("--csv", help="Mesures réelles NORMALES (sinon : simulation)")
    p.add_argument("-o", "--output", help="Chemin du .joblib (défaut : model_path)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Évalue le détecteur sur la période de test simulée")
    p.add_argument("-m", "--model", help="Chemin du .joblib (défaut : model_path)")
    p.add_argument("--figures", help="Dossier où sauvegarder les graphiques")
    p.add_argument("--report", help="Fichier JSON où écrire le rapport")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("replay", help="Rejoue une panne en flux temps réel")
    p.add_argument("-m", "--model", help="Chemin du .joblib (défaut : model_path)")
    p.add_argument("--event-index", type=int, default=0, help="Index de la panne dans la config")
    p.add_argument("--warmup", type=int, default=60, help="Minutes de flux avant la panne")
    p.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
