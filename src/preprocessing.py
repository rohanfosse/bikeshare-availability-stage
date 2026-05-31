"""Pré-traitement des données de disponibilité.

Deux idées clés :

1. **Encodage cyclique** des variables temporelles. La « minute depuis minuit »
   (MM) et le « jour de semaine » (WD) sont périodiques : minuit (0) doit être
   proche de 23h59 (1439). On les projette donc sur un cercle via sin/cos plutôt
   que de les passer en valeurs brutes.

2. **Normalisation par la capacité.** Le nombre de vélos disponibles (AB) est
   divisé par la capacité de la station, ce qui donne un *taux d'occupation*
   dans [0, 1] et permet de comparer des stations de tailles différentes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode_cyclic(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode une variable périodique en deux composantes (sin, cos).

    Args:
        values: valeurs brutes (ex. minutes depuis minuit).
        period: période du cycle (ex. 1440 pour la journée, 7 pour la semaine).

    Returns:
        (sin, cos) de même longueur que ``values``.
    """
    angle = 2.0 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_features(
    df: pd.DataFrame,
    capacity: int,
    use_weather: bool = False,
) -> pd.DataFrame:
    """Construit la matrice de features à partir d'un journal minute par minute.

    Le DataFrame d'entrée doit contenir les colonnes :
        - ``timestamp`` (datetime)
        - ``available_bikes`` (int)  -> AB
        - ``temperature`` (float)    -> T   (si use_weather)
        - ``humidity`` (float)       -> RH  (si use_weather)

    Renvoie un DataFrame de features prêtes pour les modèles :
        - occupancy            : AB normalisé par la capacité (cible et entrée)
        - mm_sin, mm_cos       : minute depuis minuit, encodée cycliquement
        - wd_sin, wd_cos       : jour de semaine, encodé cycliquement
        - (t_norm, rh_norm)    : météo normalisée, si use_weather

    Sans météo on a 5 colonnes de features (occupancy + mm_sin/cos + wd_sin/cos),
    et 2 de plus (t_norm, rh_norm) avec la météo.
    """
    out = pd.DataFrame(index=df.index)

    # Cible / entrée principale : taux d'occupation dans [0, 1].
    out["occupancy"] = df["available_bikes"].to_numpy() / float(capacity)

    ts = pd.to_datetime(df["timestamp"])
    minutes = ts.dt.hour * 60 + ts.dt.minute           # MM : 0..1439
    weekday = ts.dt.weekday                            # WD : 0 (lundi) .. 6

    out["mm_sin"], out["mm_cos"] = encode_cyclic(minutes.to_numpy(), period=1440)
    out["wd_sin"], out["wd_cos"] = encode_cyclic(weekday.to_numpy(), period=7)

    if use_weather:
        # Normalisation min-max simple, suffisante pour démarrer.
        out["t_norm"] = _minmax(df["temperature"].to_numpy())
        out["rh_norm"] = _minmax(df["humidity"].to_numpy())

    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def make_windows(
    features: pd.DataFrame,
    input_steps: int = 60,
    horizon: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe la série en fenêtres glissantes (apprentissage supervisé).

    Pour chaque instant t, on prend ``input_steps`` minutes d'historique en
    entrée et on cherche à prévoir les ``horizon`` minutes suivantes du taux
    d'occupation.

    Returns:
        X de forme (n, input_steps, n_features) : séquences d'entrée (LSTM).
        y de forme (n, horizon) : taux d'occupation à prévoir.

    Note : pour le MLP, on aplatit X en (n, input_steps * n_features) dans le
    module du modèle.
    """
    mat = features.to_numpy(dtype=np.float32)
    occ = features["occupancy"].to_numpy(dtype=np.float32)
    n_features = mat.shape[1]

    xs, ys = [], []
    last = len(mat) - input_steps - horizon
    for t in range(last + 1):
        xs.append(mat[t : t + input_steps])
        ys.append(occ[t + input_steps : t + input_steps + horizon])

    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return X.reshape(-1, input_steps, n_features), y
