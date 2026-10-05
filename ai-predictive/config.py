import os
from dotenv import load_dotenv

load_dotenv()

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER") or None
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD") or None
TOPIC_SENSORS = os.getenv("TOPIC_SENSORS", "sentinel/sensors")
TOPIC_ANOMALY = os.getenv("TOPIC_ANOMALY", "sentinel/ai/anomaly")
API_URL = os.getenv("API_URL", "http://localhost:3000/api/v1/alerts")