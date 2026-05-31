"""MLP (Multi-Layer Perceptron) : réseau de neurones dense.

Un MLP enchaîne des couches de neurones entièrement connectées. C'est un modèle
simple et rapide, capable d'apprendre des relations non linéaires entre les
entrées et la sortie.

Architecture choisie ici :
    - 2 couches denses cachées de 100 neurones, activation ReLU
    - 1 couche de sortie linéaire de dimension 60 (prévision minute par minute)
    - optimiseur Adam (lr 0.001), perte MSE, batch 32, early stopping (patience 10)

Le MLP ne modélise pas la dimension temporelle de la séquence : on aplatit donc
la fenêtre d'entrée (input_steps × n_features) en un seul vecteur.
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(input_dim: int, horizon: int = 60) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(100, activation="relu", kernel_initializer="glorot_uniform"),
            layers.Dense(100, activation="relu", kernel_initializer="glorot_uniform"),
            layers.Dense(horizon, activation="linear"),
        ],
        name="mlp",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_mlp(X_train, y_train, X_val, y_val, epochs: int = 300, batch_size: int = 32):
    """Entraîne le MLP. X_* est attendu en (n, steps, features) puis aplati."""
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)

    model = build_mlp(input_dim=X_train_flat.shape[1], horizon=y_train.shape[1])
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train_flat,
        y_train,
        validation_data=(X_val_flat, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=0,
    )
    return model, history


def predict_mlp(model: keras.Model, X) -> np.ndarray:
    return model.predict(X.reshape(len(X), -1), verbose=0)
