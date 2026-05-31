"""LSTM (Long Short-Term Memory) pour la prévision de séquences.

Un LSTM est un réseau de neurones récurrent capable de retenir une information
sur plusieurs pas de temps grâce à une cellule de mémoire et des "portes"
(entrée, oubli, sortie). C'est ce qui le rend adapté aux séries temporelles : il
apprend les dépendances temporelles (cycles, tendances) de l'historique.

Architecture choisie ici :
    - 2 couches LSTM empilées de 100 unités
    - dropout 0.2 entre les couches (régularisation : on "éteint" aléatoirement
      des neurones à l'entraînement pour limiter le surapprentissage)
    - initialisation Xavier/Glorot (garde des gradients équilibrés au démarrage)
    - couche de sortie linéaire de dimension 60 (prévision minute par minute)
    - optimiseur Adam (lr 0.001), perte MSE, batch 32, early stopping (patience 10)

Contrairement au MLP, le LSTM consomme directement la séquence
(input_steps × n_features) sans l'aplatir.
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm(input_steps: int, n_features: int, horizon: int = 60) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(input_steps, n_features)),
            layers.LSTM(
                100,
                return_sequences=True,            # renvoie la séquence à la couche suivante
                kernel_initializer="glorot_uniform",
            ),
            layers.Dropout(0.2),
            layers.LSTM(100, kernel_initializer="glorot_uniform"),
            layers.Dropout(0.2),
            layers.Dense(horizon, activation="linear"),
        ],
        name="lstm",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_lstm(X_train, y_train, X_val, y_val, epochs: int = 300, batch_size: int = 32):
    model = build_lstm(
        input_steps=X_train.shape[1],
        n_features=X_train.shape[2],
        horizon=y_train.shape[1],
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=0,
    )
    return model, history


def predict_lstm(model: keras.Model, X) -> np.ndarray:
    return model.predict(X, verbose=0)
