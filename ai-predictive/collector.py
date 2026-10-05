"""
Collecteur de mesures Sentinel-X.

Ce programme s'abonne au topic MQTT des capteurs et enregistre chaque mesure
reçue dans un fichier CSV (un nouveau fichier à chaque lancement).

Il ne fait aucune différence entre le vrai ESP8266 et le simulateur :
il enregistre tout ce qui arrive sur le topic.

Exemples d'utilisation :
  python collector.py                    -> fichier data/raw/session_AAAAMMJJ_HHMMSS.csv
  python collector.py --name salle_nuit  -> fichier data/raw/salle_nuit_AAAAMMJJ_HHMMSS.csv
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

import config

# Colonnes du fichier CSV, dans l'ordre
COLUMNS = ["timestamp", "device", "temp", "hum", "gas", "pir"]

# Dossier où ranger les fichiers (data/raw à côté de ce script)
RAW_DIR = Path(__file__).parent / "data" / "raw"


def parse_args():
    parser = argparse.ArgumentParser(description="Collecteur de mesures Sentinel-X")
    parser.add_argument("--name", default="session",
                        help="début du nom du fichier, pour t'y retrouver (ex: salle_nuit, sim_combined)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Préparer le fichier CSV
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = RAW_DIR / f"{args.name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    csv_file = open(filename, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
    writer.writeheader()

    stats = {"ok": 0, "rejected": 0}

    # 2. Ce qui se passe quand on se connecte au broker
    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"Connexion refusée par le broker : {reason_code}")
            return
        # On s'abonne ici : comme ça, si la connexion saute puis revient, on se réabonne tout seul
        client.subscribe(config.TOPIC_SENSORS)
        print(f"Connecté. Écoute du topic '{config.TOPIC_SENSORS}'")
        print(f"Enregistrement dans : {filename}  (Ctrl+C pour arrêter)\n")

    # 3. Ce qui se passe à chaque message reçu
    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["rejected"] += 1
            print(f"Message illisible ignoré : {msg.payload[:80]!r}")
            return

        # C'est le serveur qui horodate (l'ESP8266 n'a pas d'horloge fiable)
        row = {"timestamp": datetime.now().isoformat(timespec="milliseconds")}
        for col in COLUMNS[1:]:
            row[col] = data.get(col)    # None si le champ manque (ex: lecture DHT22 ratée)

        writer.writerow(row)
        csv_file.flush()                # écrit tout de suite sur le disque : rien n'est perdu si ça plante
        stats["ok"] += 1

        if stats["ok"] % 10 == 0:
            print(f"{stats['ok']} mesures enregistrées | dernière : {row}")

    # 4. Ce qui se passe si la connexion saute
    def on_disconnect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"Connexion perdue ({reason_code}), tentative de reconnexion automatique...")

    # 5. Connexion au broker (syntaxe paho-mqtt version 2)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ai-collector")
    if config.MQTT_USER:
        client.username_pw_set(config.MQTT_USER, config.MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        client.connect(config.MQTT_HOST, config.MQTT_PORT)
    except (ConnectionRefusedError, OSError):
        print(f"ERREUR : impossible de joindre le broker MQTT sur {config.MQTT_HOST}:{config.MQTT_PORT}.")
        print("Vérifie que Mosquitto est bien lancé, et que MQTT_HOST / MQTT_PORT sont corrects dans .env")
        csv_file.close()
        filename.unlink()               # on supprime le fichier vide
        return

    # 6. Écouter en boucle jusqu'au Ctrl+C
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        csv_file.close()
        print(f"\nArrêt. {stats['ok']} mesures enregistrées, {stats['rejected']} messages ignorés.")
        print(f"Fichier : {filename}")


if __name__ == "__main__":
    main()