"""Référence XGBoost (gradient boosting d'arbres de décision).

XGBoost construit une suite d'arbres de décision où chaque nouvel arbre corrige
les erreurs des précédents. C'est une référence très solide pour la régression
sur données tabulaires : rapide, robuste, peu de réglages.

Particularité ici : XGBoost ne prédit qu'une seule valeur par modèle. Pour
prévoir les 60 minutes à venir, on enveloppe le régresseur dans un
``MultiOutputRegressor`` (un arbre par horizon de minute).

Réglages : 200 arbres, profondeur max 6, learning rate 0.1.
"""

from __future__ import annotations

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def build_xgboost() -> MultiOutputRegressor:
    base = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    return MultiOutputRegressor(base)


def train_xgboost(X_train, y_train):
    """X_train en (n, steps, features) : on aplatit pour un modèle tabulaire."""
    model = build_xgboost()
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model


def predict_xgboost(model, X) -> np.ndarray:
    return model.predict(X.reshape(len(X), -1))
