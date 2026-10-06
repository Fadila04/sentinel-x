# Sentinel-X — Maintenance prédictive

Détection précoce d'anomalies sur les capteurs du boîtier ESP8266 (DHT22, MQ-2) avec un
**Isolation Forest non supervisé** : le modèle apprend le fonctionnement normal pendant une
période de calibration, puis signale toute dérive *avant* qu'un seuil critique soit atteint.
Aucune règle du type `if temp > 40` n'est utilisée par le modèle.

## Structure

```
predictive_maintenance/
├── configs/default.yaml          # tous les hyperparamètres (aucune constante dans le code)
├── src/sentinel_pm/
│   ├── config.py                 # dataclasses de configuration + chargement YAML
│   ├── data/
│   │   ├── simulation.py         # simulateur capteurs + injection de pannes
│   │   └── loaders.py            # chargement de mesures réelles (CSV)
│   ├── features.py               # feature engineering — source de vérité unique
│   ├── model.py                  # entraînement, seuil, persistance, sauvegarde
│   ├── evaluation.py             # métriques par mesure et par événement
│   ├── realtime.py               # détecteur temps réel + contrat du payload d'alerte
│   ├── visualization.py          # graphiques (optionnel)
│   └── cli.py                    # commande `sentinel-pm`
├── tests/                        # unitaires + non-régression (29 tests)
├── notebooks/
│   ├── 00_prototype_isolation_forest.ipynb   # prototype d'origine (archive, non maintenu)
│   └── 01_isolation_forest.ipynb             # démarche expliquée, s'appuie sur le package
├── models/                       # détecteurs entraînés (.joblib, non versionnés)
└── reports/                      # figures et rapports d'évaluation (non versionnés)
```

## Installation

```bash
cd predictive_maintenance
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # ou ".[viz]" / ".[notebook]" selon le besoin
```

## Utilisation

```bash
make train      # entraîne sur la calibration simulée -> models/sentinel_isolation_forest.joblib
make evaluate   # métriques + rapport JSON + figures dans reports/
make replay     # rejoue une dérive lente mesure par mesure et affiche le payload d'alerte
make test       # tests ; `make test-fast` pour sauter la non-régression
```

Avec de vraies mesures (fonctionnement normal uniquement) :

```bash
sentinel-pm -c configs/default.yaml train --csv data/calibration_esp8266.csv
```

Format attendu : `timestamp,temperature,humidity,gas` (colonnes supplémentaires ignorées).

En Python, côté service qui reçoit les mesures :

```python
from sentinel_pm import load_bundle, RealTimeDetector

detector = RealTimeDetector(load_bundle("models/sentinel_isolation_forest.joblib"), device_id="esp8266-01")
alert = detector.update({"timestamp": ts, "temperature": t, "humidity": h, "gas": g})
if alert:
    post("/api/v1/alerts", json=alert)
```

## Méthode

1. **Features cinétiques** sur fenêtres glissantes (15 et 30 mesures) : pentes, écarts-types et
   amplitudes (échelle log), corrélation température/gaz. Le même code sert à l'entraînement
   et en temps réel.
2. **Isolation Forest** entraîné uniquement sur des données normales.
3. **Seuil appris** : 0,5ᵉ percentile des scores en fonctionnement normal.
4. **Persistance** : alerte après 3 mesures consécutives sous le seuil.

## Résultats de référence (config par défaut, données simulées)

| Indicateur | Valeur |
|---|---|
| Précision / rappel / F1 (par mesure) | 0,897 / 0,787 / 0,839 |
| Pannes détectées | 5 / 5 |
| Délai de détection | 5 à 33 min selon la panne |
| Avance sur un seuil fixe à 38 °C (dérives lentes) | 139 et 166 min |
| Fausses alertes | 50 / 2295 (49 effet de queue, 1 spontanée) |

Ces valeurs sont figées par `tests/test_regression.py`.

| `threshold_quantile` | Précision | Rappel | F1 | Faux positifs |
|---|---|---|---|---|
| 0.001 | 0,917 | 0,701 | 0,795 | 35 |
| **0.005** (défaut) | 0,897 | 0,787 | 0,839 | 50 |
| 0.02 | 0,845 | 0,823 | 0,834 | 84 |

## Contrat d'alerte (à valider avec l'équipe DEV)

```json
{
  "device_id": "esp8266-01",
  "type": "predictive_anomaly",
  "severity": "warning | critical",
  "score": -0.0775,
  "timestamp": "2026-10-09T13:32:00",
  "measures": {"temperature": 26.53, "humidity": 53.86, "gas": 185.72}
}
```

## Limites connues et prochaines étapes

- Les fenêtres sont exprimées en **nombre de mesures** et calibrées pour 1 mesure/minute. Avec
  une ESP8266 qui envoie toutes les 2 à 10 s, il faudra ajuster `window` / `window_long` (ou
  ré-échantillonner les mesures à la minute) avant l'entraînement.
- Le modèle doit être **ré-entraîné sur une calibration réelle** du boîtier ; les résultats
  ci-dessus valident la méthode, pas les performances terrain.
- Un `.joblib` est lié à la version de scikit-learn utilisée (un avertissement est émis en cas
  d'écart) : ré-entraîner après une mise à jour.
