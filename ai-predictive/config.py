"""
Réglages du module IA, lus depuis le fichier .env (jamais de secret dans le code).
"""

import os

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# --- MQTT ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER") or None
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD") or None
# Chemin du certificat de l'autorité (fourni par la CYBER). Vide = pas de TLS.
MQTT_CA_CERT = os.getenv("MQTT_CA_CERT") or None

TOPIC_SENSORS = os.getenv("TOPIC_SENSORS", "sentinel/sensors")
TOPIC_ANOMALY = os.getenv("TOPIC_ANOMALY", "sentinel/ai/anomaly")

# --- API des DEV ---
API_URL = os.getenv("API_URL", "http://localhost:3000/api/v1/alerts")
API_TOKEN = os.getenv("API_TOKEN") or None   # si l'API demande une authentification


def create_mqtt_client(client_id):
    """Crée un client MQTT avec identifiants et TLS si ils sont configurés dans .env."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    if MQTT_CA_CERT:
        client.tls_set(ca_certs=MQTT_CA_CERT)   # active le chiffrement (MQTTS)
    return client