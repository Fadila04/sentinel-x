"""
Fausse API pour tester le service IA sans l'API des DEV.

Elle écoute sur http://localhost:3000 et affiche chaque alerte reçue sur /api/v1/alerts.

Utilisation :
  python fake_api.py
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 3000


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/api/v1/alerts":
            self.send_response(404)
            self.end_headers()
            return
        alert = json.loads(body)
        print(f"\n=== ALERTE REÇUE ({alert.get('event')}) ===")
        print(json.dumps(alert, indent=2, ensure_ascii=False))
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "received"}')

    def log_message(self, *args):
        pass   # pas de log technique à chaque requête


if __name__ == "__main__":
    print(f"Fausse API en écoute sur http://localhost:{PORT}/api/v1/alerts  (Ctrl+C pour arrêter)")
    HTTPServer(("localhost", PORT), Handler).serve_forever()