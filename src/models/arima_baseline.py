"""Référence ARIMA (modèle statistique de série temporelle).

ARIMA(p, d, q) modélise une série à partir d'elle-même :
    - AR (p) : combinaison linéaire des p valeurs passées,
    - I  (d) : nombre de différenciations pour rendre la série stationnaire,
    - MA (q) : combinaison des q erreurs passées.

C'est une référence linéaire purement univariée : elle n'utilise que
l'historique du nombre de vélos, sans variables annexes (météo, jour…). Ici on
prend (p, d, q) = (2, 0, 2).

Pour prévoir l'heure à venir on fait une **prévision glissante** (rolling) :
à chaque pas on réajuste le modèle sur l'historique disponible et on prévoit les
60 minutes suivantes. C'est plus lent que les réseaux de neurones, d'où son rôle
de simple point de comparaison.
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

ORDER = (2, 0, 2)


def forecast_arima(history: np.ndarray, horizon: int = 60, order=ORDER) -> np.ndarray:
    """Ajuste un ARIMA sur ``history`` et prévoit ``horizon`` pas en avant."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # ARIMA est bavard sur la convergence
        model = ARIMA(history, order=order).fit()
        return np.asarray(model.forecast(steps=horizon))


def rolling_forecast(series: np.ndarray, starts, horizon: int = 60, order=ORDER) -> np.ndarray:
    """Prévision glissante sur plusieurs points de départ.

    Args:
        series: série complète du taux d'occupation.
        starts: indices à partir desquels prévoir (l'historique = series[:start]).
        horizon: nombre de minutes à prévoir.

    Returns:
        Tableau (len(starts), horizon) des prévisions.
    """
    preds = []
    for s in starts:
        preds.append(forecast_arima(series[:s], horizon=horizon, order=order))
    return np.asarray(preds)
