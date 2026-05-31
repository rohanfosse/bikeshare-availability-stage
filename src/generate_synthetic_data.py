"""Génère une station synthétique réaliste pour démarrer sans jeu de données réel.

On reproduit grossièrement un profil d'usage urbain typique :
    - forte disponibilité la nuit (peu d'usage),
    - trois creux d'usage en semaine : matin (~8h-9h), midi (~12h-14h), soir
      (~17h-20h, le plus marqué),
    - profil plus lisse le week-end.

La cible "available_bikes" oscille donc entre ~capacité (nuit) et ~quelques
vélos (heures de pointe), avec du bruit. Ce n'est PAS une vraie donnée, mais ça
suffit pour faire tourner et déboguer toute la chaîne.

Usage :
    python -m src.generate_synthetic_data        # écrit data/synthetic_station.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAPACITY = 22  # capacité (nombre de bornes) de la station type


def _daily_usage_profile(minutes: np.ndarray, weekend: bool) -> np.ndarray:
    """Part de vélos *en circulation* (0..1) au cours de la journée."""
    hour = minutes / 60.0
    if weekend:
        # Montée lente, pic vers 19h, rien tôt le matin.
        return 0.45 * np.exp(-((hour - 19.0) ** 2) / (2 * 3.0 ** 2))
    # Semaine : trois gaussiennes (matin, midi, soir).
    morning = 0.30 * np.exp(-((hour - 8.5) ** 2) / (2 * 0.8 ** 2))
    noon = 0.30 * np.exp(-((hour - 13.0) ** 2) / (2 * 1.2 ** 2))
    evening = 0.55 * np.exp(-((hour - 18.5) ** 2) / (2 * 1.5 ** 2))
    return morning + noon + evening


def generate(
    start: str = "2024-09-01",
    days: int = 90,
    capacity: int = CAPACITY,
    seed: int = 42,
) -> pd.DataFrame:
    """Construit un journal minute par minute sur ``days`` jours."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=days * 1440, freq="min")

    minute_of_day = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    is_weekend = index.weekday.to_numpy() >= 5

    usage = np.where(
        is_weekend,
        _daily_usage_profile(minute_of_day, weekend=True),
        _daily_usage_profile(minute_of_day, weekend=False),
    )
    # Vélos disponibles = capacité * (1 - usage) + bruit.
    available = capacity * (1.0 - usage)
    available += rng.normal(0.0, 1.0, size=available.shape)          # bruit capteur
    available = np.clip(np.rint(available), 0, capacity).astype(int)

    # Météo synthétique (cycle saisonnier + journalier léger).
    t = np.arange(len(index))
    temperature = (
        15.0
        + 8.0 * np.sin(2 * np.pi * t / (days * 1440))   # saison
        + 4.0 * np.sin(2 * np.pi * minute_of_day / 1440)  # jour/nuit
        + rng.normal(0.0, 1.0, size=len(index))
    )
    humidity = np.clip(
        65.0 - 0.8 * (temperature - 15.0) + rng.normal(0.0, 5.0, size=len(index)),
        15.0,
        100.0,
    )

    return pd.DataFrame(
        {
            "timestamp": index,
            "available_bikes": available,
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
        }
    )


if __name__ == "__main__":
    import pathlib

    df = generate()
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "synthetic_station.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} lignes écrites dans {out}")
    print(df.head())
