"""
Score de risque et règle d'alerte Sentinel-X.

Comme features.py, ce fichier est PARTAGÉ entre l'entraînement, l'évaluation et le
service temps réel : la façon de calculer le risque et de déclencher une alerte est
ainsi exactement la même partout.
"""

import numpy as np

import features as feat


def risk_from_raw(raw_scores, tau):
    """
    Transforme le score brut d'Isolation Forest en risque de 0 à 100.
    raw_scores : sortie de decision_function (positif = normal, négatif = anormal, 0 = frontière)
    50 = exactement sur la frontière apprise par le modèle.
    """
    return 100 / (1 + np.exp(np.asarray(raw_scores) / tau))


def score(bundle, X):
    """
    Pour un tableau de features X, renvoie :
      risk   : le risque de 0 à 100 donné par le modèle d'IA
      faults : True si un capteur semble figé (diagnostic de santé)
    """
    raw = bundle["pipeline"].decision_function(X[bundle["feature_columns"]])
    risk = risk_from_raw(raw, bundle["tau"])
    faults = feat.sensor_faults(X).any(axis=1).to_numpy()
    return risk, faults


class AlertTracker:
    """
    Décide quand une alerte commence et quand elle se termine, mesure après mesure.

    - L'alerte COMMENCE quand le risque est >= alert_threshold (ou un capteur est figé)
      pendant 'consecutive' mesures d'affilée. Un pic isolé ne déclenche donc rien.
    - L'alerte SE TERMINE quand le risque est < clear_threshold (et plus aucun capteur figé)
      pendant 'consecutive' mesures d'affilée. Le seuil de fin est plus bas que celui de début
      (hystérésis) : ça évite que l'alerte clignote quand le risque tourne autour de 50.
    """

    def __init__(self, alert_threshold=50, clear_threshold=40, consecutive=5):
        self.alert_threshold = alert_threshold
        self.clear_threshold = clear_threshold
        self.consecutive = consecutive
        self.active = False
        self._count = 0

    @classmethod
    def from_bundle(cls, bundle):
        """Crée le tracker avec les réglages enregistrés dans le modèle."""
        return cls(bundle["alert_threshold"], bundle["clear_threshold"], bundle["consecutive"])

    def update(self, risk, fault=False):
        """Donne une nouvelle mesure. Renvoie "started", "cleared" ou None (pas de changement)."""
        if not self.active:
            high = risk >= self.alert_threshold or fault
            self._count = self._count + 1 if high else 0
            if self._count >= self.consecutive:
                self.active, self._count = True, 0
                return "started"
        else:
            low = risk < self.clear_threshold and not fault
            self._count = self._count + 1 if low else 0
            if self._count >= self.consecutive:
                self.active, self._count = False, 0
                return "cleared"
        return None