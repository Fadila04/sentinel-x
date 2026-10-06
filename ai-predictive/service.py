"""
Service temps réel Sentinel-X : maintenance prédictive.

Tourne en continu pendant la démo :
  1. écoute les mesures des capteurs sur MQTT ;
  2. garde les 5 dernières minutes en mémoire ;
  3. calcule les features, le risque (0-100) et le diagnostic des capteurs ;
  4. publie l'état à CHAQUE mesure sur le topic IA (pour la courbe du dashboard) ;
  5. envoie un POST à l'API des DEV quand une alerte COMMENCE ou SE TERMINE.

Au démarrage, il lui faut 5 minutes de mesures avant de donner un premier risque
(statut "calibrating"). Lance-le donc au moins 5 minutes avant la démo.

Utilisation :
  python service.py
"""

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import requests

import config
import features as feat
from alerting import AlertTracker, score

MODEL_PATH = Path(__file__).parent / "models" / "anomaly_model.joblib"
MAX_GAP_S = 60     # si aucune mesure pendant 60 s (boîtier redémarré...), on repart de zéro

# Noms lisibles des features, pour expliquer POURQUOI le modèle s'inquiète.
# Pour chaque feature : (texte si la valeur est trop HAUTE, texte si elle est trop BASSE)
FEATURE_LABELS = {
    "temp_mean": ("température élevée", "température basse"),
    "temp_std": ("température instable", "température figée"),
    "temp_slope": ("température en hausse", "température en baisse"),
    "temp_drift": ("température au-dessus de l'habitude", "température sous l'habitude"),
    "hum_mean": ("humidité élevée", "humidité basse"),
    "hum_std": ("humidité instable", "humidité figée"),
    "hum_slope": ("humidité en hausse", "humidité en baisse"),
    "hum_drift": ("humidité au-dessus de l'habitude", "humidité sous l'habitude"),
    "gas_mean": ("gaz élevé", "gaz bas"),
    "gas_std": ("gaz instable", "gaz figé"),
    "gas_slope": ("gaz en hausse", "gaz en baisse"),
    "gas_drift": ("gaz au-dessus de l'habitude", "gaz sous l'habitude"),
    "corr_temp_gas": ("température et gaz liés", "température et gaz opposés"),
    "slope_product": ("température et gaz montent ensemble", "température et gaz évoluent en sens inverse"),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sentinel-ai")


class PredictiveService:
    def __init__(self, bundle):
        self.bundle = bundle
        self.scaler = bundle["pipeline"].named_steps["scaler"]
        self.history = deque(maxlen=feat.BUFFER_SIZE)
        self.tracker = AlertTracker.from_bundle(bundle)
        self.last_time = None
        self.client = config.create_mqtt_client("ai-predictive-service")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    # ------------------------------------------------------------------ MQTT
    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("Connexion refusée par le broker : %s", reason_code)
            return
        client.subscribe(config.TOPIC_SENSORS)
        log.info("Connecté. Écoute de '%s', publication sur '%s'", config.TOPIC_SENSORS, config.TOPIC_ANOMALY)

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.warning("Connexion perdue (%s), reconnexion automatique...", reason_code)

    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            self.process(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Message illisible ignoré : %r", msg.payload[:80])
        except Exception:
            # Quoi qu'il arrive, le service ne doit jamais s'arrêter pendant la démo
            log.exception("Erreur pendant le traitement d'une mesure")

    # ------------------------------------------------------------ Traitement
    def process(self, data):
        now = datetime.now()

        # Trou dans les mesures : l'historique n'a plus de sens, on recommence
        if self.last_time and (now - self.last_time).total_seconds() > MAX_GAP_S:
            log.warning("Aucune mesure depuis plus de %d s : remise à zéro de l'historique", MAX_GAP_S)
            self.history.clear()
            self.tracker = AlertTracker.from_bundle(self.bundle)
        self.last_time = now

        self.history.append({col: data.get(col) for col in ["temp", "hum", "gas", "pir"]})
        device = data.get("device", "inconnu")
        base = {"timestamp": now.isoformat(timespec="seconds"), "device": device}

        X = feat.features_from_history(self.history)
        if X is None:
            progress = min(len(self.history) / feat.REQUIRED_HISTORY, 1.0)
            self.publish({**base, "status": "calibrating", "progress": round(progress, 2),
                          "risk": None, "alert": False, "event": None})
            if len(self.history) % 15 == 0:
                log.info("Calibration : %d / %d mesures", len(self.history), feat.REQUIRED_HISTORY)
            return

        risk, faults = score(self.bundle, X)
        risk, fault = float(risk[0]), bool(faults[0])
        fault_detail = {k: bool(v) for k, v in feat.sensor_faults(X).iloc[0].items()}
        reasons = self.explain(X)

        event = self.tracker.update(risk, fault)
        status = "alert" if self.tracker.active else "ok"
        message = {**base, "status": status, "risk": round(risk), "alert": self.tracker.active,
                   "event": event, "sensor_faults": fault_detail, "reasons": reasons}
        self.publish(message)

        log.info("risque %3d/100  %-5s %s", round(risk), status.upper(),
                 ("| " + ", ".join(reasons)) if reasons else "")
        if event:
            self.send_alert(message)

    def explain(self, X, top=2, min_z=3.0):
        """Les features les plus éloignées de la normale (en nombre d'écarts-types)."""
        z = dict(zip(self.bundle["feature_columns"],
                     self.scaler.transform(X[self.bundle["feature_columns"]])[0]))
        # "montent ensemble" n'est vrai que si les DEUX pentes sont nettement positives
        if not (z["temp_slope"] > 2 and z["gas_slope"] > 2):
            z.pop("slope_product")
        ranked = sorted(z.items(), key=lambda p: -abs(p[1]))
        return [FEATURE_LABELS[name][0 if value > 0 else 1] for name, value in ranked[:top] if abs(value) >= min_z]

    # ---------------------------------------------------------------- Sorties
    def publish(self, message):
        self.client.publish(config.TOPIC_ANOMALY, json.dumps(message, ensure_ascii=False))

    def send_alert(self, message):
        """Prévient l'API des DEV. Lancé à part pour ne jamais bloquer la réception des mesures."""
        faults = [name for name, on in message["sensor_faults"].items() if on]
        if message["event"] == "started":
            kind = "sensor_fault" if faults else "predictive_anomaly"
            text = (f"Capteur en panne : {', '.join(faults)}" if kind == "sensor_fault"
                    else f"Anomalie prédictive (risque {message['risk']}/100) : {', '.join(message['reasons']) or 'comportement inhabituel'}")
            log.warning(">>> ALERTE : %s", text)
        else:
            kind, text = "predictive_anomaly_cleared", "Retour à la normale"
            log.info(">>> FIN D'ALERTE : %s", text)

        payload = {"source": "ai-predictive", "type": kind, "event": message["event"],
                   "severity": "critical" if message["event"] == "started" else "info",
                   "risk": message["risk"], "message": text, "device": message["device"],
                   "timestamp": message["timestamp"], "reasons": message["reasons"]}
        threading.Thread(target=self._post, args=(payload,), daemon=True).start()

    def _post(self, payload):
        headers = {"Authorization": f"Bearer {config.API_TOKEN}"} if config.API_TOKEN else {}
        try:
            r = requests.post(config.API_URL, json=payload, headers=headers, timeout=3)
            if r.ok:
                log.info("API prévenue (%s)", r.status_code)
            else:
                log.error("L'API a répondu %s : %s", r.status_code, r.text[:200])
        except requests.RequestException as e:
            log.error("Impossible de joindre l'API (%s) : %s", config.API_URL, e.__class__.__name__)

    # ------------------------------------------------------------- Démarrage
    def run(self):
        try:
            self.client.connect(config.MQTT_HOST, config.MQTT_PORT)
        except (ConnectionRefusedError, OSError):
            log.error("Impossible de joindre le broker MQTT sur %s:%s. Mosquitto est-il lancé ?",
                      config.MQTT_HOST, config.MQTT_PORT)
            return
        log.info("Calibration : il faut %d mesures (~%d min) avant le premier score",
                 feat.REQUIRED_HISTORY, feat.REQUIRED_HISTORY * 2 // 60)
        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            log.info("Arrêt du service.")
        finally:
            self.client.disconnect()


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Modèle introuvable. Lance d'abord : python train.py")
    bundle = joblib.load(MODEL_PATH)
    log.info("Modèle chargé (%d features, alerte après %d mesures au-dessus de %d)",
             len(bundle["feature_columns"]), bundle["consecutive"], bundle["alert_threshold"])
    PredictiveService(bundle).run()


if __name__ == "__main__":
    main()