"""Métriques d'évaluation : MAE et RMSE.

On évalue toujours sur le **nombre réel de vélos** (entier), pas sur le taux
d'occupation normalisé. La fonction ``rescale`` permet de repasser du taux
([0, 1]) au nombre de vélos en multipliant par la capacité.

- MAE  : écart moyen, en vélos. Facile à interpréter ("on se trompe de X vélos").
- RMSE : pénalise davantage les grosses erreurs. RMSE >= MAE toujours ; un grand
         écart entre les deux signale des erreurs ponctuelles importantes.

On présente en général les deux conjointement : la MAE donne l'erreur typique,
la RMSE révèle la présence de gros écarts ponctuels.
"""

from __future__ import annotations

import numpy as np


def rescale(occupancy: np.ndarray, capacity: int) -> np.ndarray:
    """Repasse d'un taux d'occupation [0, 1] au nombre de vélos."""
    return np.asarray(occupancy, dtype=float) * float(capacity)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff ** 2)))


def errors_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons=(15, 30, 45, 60),
) -> dict[int, dict[str, float]]:
    """MAE / RMSE à différents horizons de prévision (en minutes).

    ``y_true`` et ``y_pred`` ont la forme (n, 60) : prévision minute par minute.
    L'horizon h évalue la colonne h-1 (la h-ième minute prévue).
    """
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        col = h - 1
        out[h] = {
            "MAE": mae(y_true[:, col], y_pred[:, col]),
            "RMSE": rmse(y_true[:, col], y_pred[:, col]),
        }
    return out
