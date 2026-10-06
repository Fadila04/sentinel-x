"""
Générateur de données synthétiques Sentinel-X.

Fabrique en quelques secondes des milliers de mesures, au même format que collector.py :
  - data/synthetic/train_normal.csv   : uniquement du "normal", pour ENTRAÎNER le modèle
  - data/synthetic/test_scenarios.csv : du normal + des incidents étiquetés, pour ÉVALUER le modèle
  - figures/synthetic_preview.png     : un graphique pour visualiser chaque scénario

Colonnes ajoutées par rapport aux vraies données :
  - label    : 0 = normal, 1 = incident en cours (sert uniquement à l'évaluation)
  - scenario : nom du scénario
  - run_id   : numéro de la série (chaque série est indépendante)

Exemples d'utilisation :
  python synthetic.py
  python synthetic.py --calibrate data/raw/salle_normal_20261006_100000.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # dessine dans un fichier, sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT_DIR = HERE / "data" / "synthetic"
FIG_DIR = HERE / "figures"

INTERVAL_S = 2                           # une mesure toutes les 2 secondes, comme le DHT22
COLUMNS = ["timestamp", "device", "temp", "hum", "gas", "pir", "label", "scenario", "run_id"]
SCENARIOS = ["overheat", "gas_leak", "combined", "spike", "stuck"]

# Valeurs par défaut de la salle. Remplacées automatiquement si tu utilises --calibrate.
PARAMS = {
    "temp_base": 22.5, "temp_noise": 0.1,
    "hum_base": 45.0, "hum_noise": 0.3,
    "gas_base": 300.0, "gas_noise": 4.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Générateur de données synthétiques Sentinel-X")
    parser.add_argument("--calibrate", help="CSV de vraies mesures pour régler les valeurs de base et le bruit")
    parser.add_argument("--train-hours", type=float, default=12, help="heures de données normales pour l'entraînement")
    parser.add_argument("--runs", type=int, default=5, help="nombre de séries par scénario dans le jeu de test")
    parser.add_argument("--seed", type=int, default=42, help="graine du hasard (même graine = mêmes données)")
    return parser.parse_args()


def calibrate(csv_path):
    """Lit de vraies mesures et en déduit les valeurs de base et le niveau de bruit."""
    df = pd.read_csv(csv_path).dropna(subset=["temp", "hum", "gas"])
    if len(df) < 100:
        raise SystemExit(f"Pas assez de mesures dans {csv_path} ({len(df)}), il en faut au moins 100.")
    for col in ["temp", "hum", "gas"]:
        PARAMS[f"{col}_base"] = float(df[col].median())
        # Bruit = agitation d'une mesure à la suivante (ignore les lentes variations de la salle)
        PARAMS[f"{col}_noise"] = max(float(df[col].diff().std() / np.sqrt(2)), 0.01)
    print(f"Calibré sur {len(df)} vraies mesures : " + ", ".join(f"{k}={v:.2f}" for k, v in PARAMS.items()))


def normal_series(n, rng):
    """Génère n mesures normales : base légèrement variable + lente oscillation + bruit."""
    t = np.arange(n)
    temp_base = PARAMS["temp_base"] + rng.uniform(-1, 1)          # chaque série démarre un peu différemment
    gas_base = PARAMS["gas_base"] + rng.uniform(-15, 15)
    period = rng.uniform(1800, 5400)                              # oscillation lente (1h à 3h)
    phase = rng.uniform(0, 2 * np.pi)

    temp = temp_base + 0.5 * np.sin(2 * np.pi * t / period + phase) + rng.normal(0, PARAMS["temp_noise"], n)
    gas = gas_base + 5 * np.sin(2 * np.pi * t / (period * 1.3) + phase) + rng.normal(0, PARAMS["gas_noise"], n)
    return temp, gas, temp_base


def apply_scenario(name, temp, gas, start, rng):
    """Injecte un incident à partir de l'indice 'start'. Renvoie temp, gas et l'étiquette."""
    n = len(temp)
    k = np.arange(n - start)              # 0, 1, 2... depuis le début de l'incident
    r = rng.uniform(0.7, 1.3)             # intensité variable d'une série à l'autre
    label = np.zeros(n, dtype=int)
    label[start:] = 1

    if name == "overheat":                # la température monte lentement
        temp[start:] += 0.05 * r * k
    elif name == "gas_leak":              # le gaz dérive lentement
        gas[start:] += 1.0 * r * k
    elif name == "combined":              # hausse lente + micro-dérive du gaz (l'exemple du sujet)
        temp[start:] += 0.005 * r * k     # environ +3 °C en 20 minutes
        gas[start:] += 0.15 * r * k       # environ +90 en 20 minutes
    elif name == "spike":                 # pic brutal de gaz pendant 10 mesures, puis retour au normal
        gas[start:start + 10] += 150 * r
        label[start + 10:] = 0
    elif name == "stuck":                 # capteur figé : plus aucune variation
        temp[start:] = temp[start]
        gas[start:] = gas[start]
    return temp, gas, label


def build_frame(temp, gas, temp_base, label, scenario, run_id, rng, stuck_from=None):
    """Assemble un tableau au format du collecteur."""
    n = len(temp)
    hum = PARAMS["hum_base"] - 0.5 * (temp - temp_base) + rng.normal(0, PARAMS["hum_noise"], n)
    if stuck_from is not None:
        hum[stuck_from:] = hum[stuck_from]

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-05 08:00:00", periods=n, freq=f"{INTERVAL_S}s"),
        "device": "SX-SIM",
        "temp": temp.round(1),
        "hum": hum.round(1),
        "gas": np.clip(gas, 0, 1023).round().astype(int),
        "pir": (rng.random(n) < 0.05).astype(int),
        "label": label,
        "scenario": scenario,
        "run_id": run_id,
    })
    # Comme le vrai DHT22 : environ 1 % de lectures ratées
    glitches = rng.random(n) < 0.01
    if stuck_from is not None:
        glitches[stuck_from:] = False     # un capteur figé ne "rate" pas ses lectures
    df.loc[glitches, ["temp", "hum"]] = np.nan
    return df


def make_train(hours, rng):
    """Jeu d'entraînement : plusieurs sessions normales, sans aucun incident."""
    n_sessions = 4
    n = int(hours * 3600 / INTERVAL_S / n_sessions)
    frames = []
    for s in range(n_sessions):
        temp, gas, temp_base = normal_series(n, rng)
        frames.append(build_frame(temp, gas, temp_base, np.zeros(n, dtype=int), "normal", s, rng))
    return pd.concat(frames, ignore_index=True)


def make_test(runs, rng):
    """Jeu de test : pour chaque scénario, des séries de 20 min normales puis 20 min d'incident."""
    n_normal, n_incident = 600, 600
    n = n_normal + n_incident
    frames, run_id = [], 0
    for name in ["normal"] + SCENARIOS:
        for _ in range(runs):
            temp, gas, temp_base = normal_series(n, rng)
            if name == "normal":
                label = np.zeros(n, dtype=int)
            else:
                temp, gas, label = apply_scenario(name, temp, gas, n_normal, rng)
            stuck_from = n_normal if name == "stuck" else None
            frames.append(build_frame(temp, gas, temp_base, label, name, run_id, rng, stuck_from))
            run_id += 1
    return pd.concat(frames, ignore_index=True)


def plot_preview(test):
    """Un graphique par scénario (première série), pour voir à quoi ressemblent les données."""
    names = ["normal"] + SCENARIOS
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.3 * len(names)), sharex=True)
    for ax, name in zip(axes, names):
        run = test[test["scenario"] == name]
        run = run[run["run_id"] == run["run_id"].min()].reset_index(drop=True)
        minutes = run.index * INTERVAL_S / 60
        ax.plot(minutes, run["temp"], color="tab:red", label="température (°C)")
        ax2 = ax.twinx()
        ax2.plot(minutes, run["gas"], color="tab:blue", alpha=0.7, label="gaz (brut)")
        if run["label"].any():
            ax.axvspan(minutes[run["label"] == 1].min(), minutes[run["label"] == 1].max(),
                       color="orange", alpha=0.15, label="incident")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("°C", color="tab:red")
        ax2.set_ylabel("gaz", color="tab:blue")
    axes[-1].set_xlabel("minutes")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "synthetic_preview.png"
    fig.savefig(path, dpi=110)
    return path


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.calibrate:
        calibrate(args.calibrate)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = make_train(args.train_hours, rng)
    train[COLUMNS].to_csv(OUT_DIR / "train_normal.csv", index=False)

    test = make_test(args.runs, rng)
    test[COLUMNS].to_csv(OUT_DIR / "test_scenarios.csv", index=False)

    fig_path = plot_preview(test)

    print(f"Entraînement : {len(train)} mesures normales -> {OUT_DIR / 'train_normal.csv'}")
    print(f"Test         : {len(test)} mesures, dont {int(test['label'].sum())} en incident "
          f"-> {OUT_DIR / 'test_scenarios.csv'}")
    print(f"Aperçu       : {fig_path}")
    print("\nRépartition du jeu de test :")
    print(test.groupby("scenario")["label"].agg(mesures="size", en_incident="sum").to_string())


if __name__ == "__main__":
    main()