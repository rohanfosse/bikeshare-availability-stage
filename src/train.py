"""Exemple bout-en-bout : prévoir la disponibilité d'une station à 1 h.

Chaîne complète :
    1. charge (ou génère) une station,
    2. construit les features (encodage cyclique + normalisation),
    3. découpe en fenêtres glissantes (60 min d'historique -> 60 min de prévision),
    4. sépare apprentissage / test dans le temps,
    5. entraîne un MLP et un LSTM,
    6. compare les erreurs (MAE / RMSE) par horizon, sur le nombre réel de vélos.

Lancer :
    python -m src.train
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from src import preprocessing, metrics
from src.generate_synthetic_data import CAPACITY, generate
from src.models import lstm, mlp

INPUT_STEPS = 60
HORIZON = 60
DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "synthetic_station.csv"


def load_data() -> tuple[pd.DataFrame, int]:
    """Charge la station synthétique (la génère si absente)."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    else:
        print("Pas de CSV trouvé : génération de données synthétiques.")
        df = generate()
    return df, CAPACITY


def temporal_split(X, y, train_frac=0.66):
    """Séparation apprentissage/test en respectant l'ordre temporel."""
    cut = int(len(X) * train_frac)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def report(name: str, y_true_bikes, y_pred_bikes):
    print(f"\n=== {name} (erreurs en nombre de vélos) ===")
    table = metrics.errors_by_horizon(y_true_bikes, y_pred_bikes)
    print(f"{'horizon':>8} | {'MAE':>6} | {'RMSE':>6}")
    for h, vals in table.items():
        print(f"{h:>6}min | {vals['MAE']:>6.2f} | {vals['RMSE']:>6.2f}")


def main():
    df, capacity = load_data()

    features = preprocessing.build_features(df, capacity=capacity, use_weather=False)
    X, y = preprocessing.make_windows(features, input_steps=INPUT_STEPS, horizon=HORIZON)

    # Petit sous-échantillonnage pour que l'exemple tourne vite sur un portable.
    if len(X) > 8000:
        idx = np.linspace(0, len(X) - 1, 8000).astype(int)
        X, y = X[idx], y[idx]

    X_tr, y_tr, X_te, y_te = temporal_split(X, y)
    # On garde 15 % de l'apprentissage pour la validation (early stopping).
    val_cut = int(len(X_tr) * 0.85)
    X_val, y_val = X_tr[val_cut:], y_tr[val_cut:]
    X_tr, y_tr = X_tr[:val_cut], y_tr[:val_cut]

    print(f"Fenêtres -> train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_te)}")
    y_true_bikes = metrics.rescale(y_te, capacity)

    # MLP
    mlp_model, _ = mlp.train_mlp(X_tr, y_tr, X_val, y_val, epochs=50)
    mlp_pred = metrics.rescale(mlp.predict_mlp(mlp_model, X_te), capacity)
    report("MLP", y_true_bikes, mlp_pred)

    # LSTM
    lstm_model, _ = lstm.train_lstm(X_tr, y_tr, X_val, y_val, epochs=50)
    lstm_pred = metrics.rescale(lstm.predict_lstm(lstm_model, X_te), capacity)
    report("LSTM", y_true_bikes, lstm_pred)

    print(
        "\nPour comparer aux références classiques (XGBoost, ARIMA), voir "
        "src/models/xgboost_baseline.py et src/models/arima_baseline.py."
    )


if __name__ == "__main__":
    main()
