# Prévision court terme de la disponibilité des vélos en libre-service

Base de code de départ pour la prévision à court terme du nombre de vélos
disponibles à une station de vélos en libre-service.

Le dépôt tourne avec des données synthétiques, sans jeu de données réel. Lance
`python -m src.train`, regarde les erreurs affichées, puis remplace les données
synthétiques par les vraies (voir [`data/README.md`](data/README.md)). Le code
est commenté pour servir de point d'entrée.

## Le problème

Quand on arrive à une station, la frustration principale est de la trouver vide
(pas de vélo à prendre) ou pleine (pas de borne pour reposer le sien). Si on
sait prédire l'état de la station 15 minutes à 1 heure à l'avance, l'usager peut
anticiper son trajet et l'opérateur peut rééquilibrer sa flotte au bon moment.

L'objectif est de prévoir, minute par minute sur l'heure à venir, le nombre de
vélos disponibles à une station, à partir de son historique récent (et
éventuellement de la météo). On compare deux réseaux de neurones légers (MLP et
LSTM) à deux références classiques (XGBoost et ARIMA).

## Les briques et où les trouver dans le code

| Brique | Notion | Fichier |
| --- | --- | --- |
| Pré-traitement | encodage cyclique de la minute/jour, normalisation par capacité | [`src/preprocessing.py`](src/preprocessing.py) |
| Métriques | MAE et RMSE sur le nombre de vélos | [`src/metrics.py`](src/metrics.py) |
| MLP | réseau dense, prévision des 60 prochaines minutes | [`src/models/mlp.py`](src/models/mlp.py) |
| LSTM | réseau récurrent pour les dépendances temporelles | [`src/models/lstm.py`](src/models/lstm.py) |
| XGBoost | référence : boosting d'arbres de décision | [`src/models/xgboost_baseline.py`](src/models/xgboost_baseline.py) |
| ARIMA | référence : modèle statistique de série temporelle | [`src/models/arima_baseline.py`](src/models/arima_baseline.py) |
| Données factices | une station réaliste (pics matin, midi, soir) | [`src/generate_synthetic_data.py`](src/generate_synthetic_data.py) |
| Exemple complet | entraîne MLP et LSTM puis compare | [`src/train.py`](src/train.py) |

## Démarrage rapide

```bash
# 1. Environnement (Python 3.10+ recommandé)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Lancer l'exemple complet sur données synthétiques
python -m src.train

# 3. Tester une brique isolée, par exemple les données synthétiques
python -m src.generate_synthetic_data
```

`src/train.py` génère une station synthétique, entraîne un MLP et un LSTM pour
une prévision à 60 minutes, puis affiche les MAE/RMSE par horizon (15, 30, 45 et
60 minutes).

## Conventions

- Variable cible : nombre de vélos disponibles (AB), normalisé par la capacité
  de la station, soit un taux d'occupation dans [0, 1].
- Entrées : AB (historique), MM (minutes depuis minuit), WD (jour de semaine),
  et optionnellement T (température) et RH (humidité).
- Sortie : vecteur de 60 valeurs (prévision minute par minute sur 1 h).
- Entraînement : Adam (lr 0.001), perte MSE, batch 32, early stopping.

## Pour apprendre les notions

Ressources accessibles (blogs, livres en ligne gratuits, docs officielles) pour
comprendre chaque brique. Elles sont aussi rangées dans
[`references.bib`](references.bib).

### Bases des réseaux de neurones

- Michael Nielsen, *Neural Networks and Deep Learning* (livre gratuit) :
  <http://neuralnetworksanddeeplearning.com/>
- 3Blue1Brown, *But what is a Neural Network?* (vidéos) :
  <https://www.3blue1brown.com/topics/neural-networks>

### LSTM et réseaux récurrents

- Christopher Olah, *Understanding LSTM Networks* :
  <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
- Andrej Karpathy, *The Unreasonable Effectiveness of RNNs* :
  <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>

### Encodage cyclique du temps (minute, jour)

- Ian London, *Encoding cyclical continuous features* :
  <https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/>

### Régularisation et optimisation

- *A Gentle Introduction to Dropout* :
  <https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/>
- Sebastian Ruder, *An overview of gradient descent optimization (dont Adam)* :
  <https://www.ruder.io/optimizing-gradient-descent/>

### XGBoost (boosting d'arbres)

- Doc officielle, *Introduction to Boosted Trees* :
  <https://xgboost.readthedocs.io/en/stable/tutorials/model.html>

### Séries temporelles et ARIMA

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (livre
  gratuit, ARIMA au chapitre 9) : <https://otexts.com/fpp3/>
- *How to Create an ARIMA Model in Python* :
  <https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/>

### Métriques (MAE, RMSE)

- *Regression Metrics for Machine Learning* :
  <https://machinelearningmastery.com/regression-metrics-for-machine-learning/>

### Prise en main TensorFlow / Keras

- Tutoriel officiel, *Time series forecasting* :
  <https://www.tensorflow.org/tutorials/structured_data/time_series>
