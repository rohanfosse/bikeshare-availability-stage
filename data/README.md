# Données

Ce dossier ne contient **aucune donnée réelle** (les CSV/Parquet sont ignorés
par git, voir `.gitignore`). Il sert à expliquer le format attendu.

## Format attendu

Un fichier par station, au pas d'**une minute**, avec au minimum :

| colonne | type | description |
| --- | --- | --- |
| `timestamp` | datetime | horodatage (1 ligne par minute) |
| `available_bikes` | int | nombre de vélos disponibles à la station |
| `temperature` | float | température en °C *(optionnel)* |
| `humidity` | float | humidité relative en % *(optionnel)* |

Il faut aussi connaître la **capacité** de la station (nombre de bornes), qui
sert à normaliser le nombre de vélos en taux d'occupation ∈ [0, 1].

## Données synthétiques (pour démarrer)

Tant qu'on n'a pas les vraies données, on génère une station factice réaliste :

```bash
python -m src.generate_synthetic_data   # écrit data/synthetic_station.csv
```

`src/train.py` la génère automatiquement si le fichier est absent.

## Brancher les vraies données

Remplacer le CSV synthétique par un vrai fichier au même format, puis ajuster
la capacité dans `src/train.py` (constante `CAPACITY`). Le reste de la chaîne
(features, fenêtres, modèles, métriques) est inchangé.
