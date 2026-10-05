"""
Simulateur de capteurs Sentinel-X.

Ce programme fait semblant d'être l'ESP8266 : il envoie de fausses mesures
(température, humidité, gaz, présence) sur MQTT, au même format que le vrai boîtier.

Il sert à :
  - développer le module IA sans attendre que le boîtier soit câblé ;
  - rejouer des incidents à volonté (surchauffe, fuite de gaz...) ;
  - servir de plan B pendant la démo si un capteur tombe en panne.

Exemples d'utilisation :
  python simulator.py                                  -> fonctionnement normal
  python simulator.py --scenario combined              -> incident combiné après 60 mesures
  python simulator.py --scenario overheat --anomaly-after 20 --interval 0.5
"""

import argparse
import json
import math
import random
import time

import paho.mqtt.client as mqtt

import config

# Valeurs "calmes" de la salle. A ajuster plus tard avec les vraies mesures.
BASE_TEMP = 22.5   # en °C
BASE_HUM = 45.0    # en %
BASE_GAS = 300     # valeur brute du MQ-2 (0 à 1023)

SCENARIOS = ["normal", "overheat", "gas_leak", "combined", "spike", "stuck"]


def parse_args():
    parser = argparse.ArgumentParser(description="Simulateur de capteurs Sentinel-X")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal",
                        help="type d'incident à simuler")
    parser.add_argument("--anomaly-after", type=int, default=60,
                        help="nombre de mesures normales avant le début de l'incident")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="secondes entre deux mesures (2.0 = comme le vrai DHT22)")
    parser.add_argument("--device", default="SX-01", help="identifiant du boîtier")
    parser.add_argument("--glitches", action="store_true",
                        help="envoie parfois une mesure ratée (comme un vrai DHT22)")
    return parser.parse_args()


def normal_values(step):
    """Génère une mesure normale : une valeur de base + une lente oscillation + du bruit."""
    temp = BASE_TEMP + 0.3 * math.sin(step / 200) + random.gauss(0, 0.1)
    gas = BASE_GAS + random.gauss(0, 4)
    return temp, gas


def apply_scenario(scenario, k, temp, gas, frozen):
    """Modifie la mesure normale selon l'incident. k = nombre de mesures depuis le début de l'incident."""
    if scenario == "overheat":      # la température monte lentement
        temp += 0.05 * k
    elif scenario == "gas_leak":    # le gaz dérive lentement
        gas += 1.0 * k
    elif scenario == "combined":    # les deux montent doucement ensemble (l'exemple du sujet)
        temp += 0.02 * k
        gas += 0.5 * k
    elif scenario == "spike":       # pic brutal de gaz pendant 10 mesures
        if k < 10:
            gas += 150
    elif scenario == "stuck":       # capteur figé : toujours exactement la même valeur
        temp, gas = frozen
    return temp, gas


def main():
    args = parse_args()

    # Connexion au broker MQTT (syntaxe de paho-mqtt version 2)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"simulator-{args.device}")
    if config.MQTT_USER:
        client.username_pw_set(config.MQTT_USER, config.MQTT_PASSWORD)
    try:
        client.connect(config.MQTT_HOST, config.MQTT_PORT)
    except (ConnectionRefusedError, OSError):
        print(f"ERREUR : impossible de joindre le broker MQTT sur {config.MQTT_HOST}:{config.MQTT_PORT}.")
        print("Vérifie que Mosquitto est bien lancé, et que MQTT_HOST / MQTT_PORT sont corrects dans .env")
        return
    client.loop_start()

    print(f"Connecté à {config.MQTT_HOST}:{config.MQTT_PORT}")
    print(f"Scénario : {args.scenario} | incident après {args.anomaly_after} mesures "
          f"| une mesure toutes les {args.interval}s")
    print(f"Envoi sur le topic : {config.TOPIC_SENSORS}  (Ctrl+C pour arrêter)\n")

    step = 0
    frozen = None
    try:
        while True:
            temp, gas = normal_values(step)

            in_anomaly = args.scenario != "normal" and step >= args.anomaly_after
            if in_anomaly:
                k = step - args.anomaly_after
                if frozen is None:
                    frozen = (temp, gas)
                temp, gas = apply_scenario(args.scenario, k, temp, gas, frozen)

            # L'humidité relative baisse quand la température monte (relation physique)
            hum = BASE_HUM - 0.5 * (temp - BASE_TEMP) + random.gauss(0, 0.3)
            if args.scenario == "stuck" and in_anomaly:
                hum = BASE_HUM

            payload = {
                "device": args.device,
                "temp": round(temp, 1),
                "hum": round(hum, 1),
                "gas": int(min(max(gas, 0), 1023)),   # le MQ-2 renvoie un entier entre 0 et 1023
                "pir": 1 if random.random() < 0.05 else 0,
            }

            # Comme un vrai DHT22, une lecture rate de temps en temps
            if args.glitches and random.random() < 0.02:
                payload["temp"] = None
                payload["hum"] = None

            client.publish(config.TOPIC_SENSORS, json.dumps(payload))
            marker = "  <-- INCIDENT" if in_anomaly else ""
            print(f"[{step:4d}] {payload}{marker}")

            step += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nArrêt du simulateur.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()