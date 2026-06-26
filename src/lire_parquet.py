"""
Lire et extraire les donnees GBFS au format Parquet.

Ce script montre, etape par etape, comment ouvrir les fichiers Parquet de
disponibilite des stations de velos en libre-service, et comment en extraire
l'information utile : une ville, une station, une periode, des statistiques.

Ou sont les donnees :
  data/status_villes/<ville>.parquet  un fichier par ville (40 villes)
  data/status_villes/index.csv        recapitulatif (lignes, stations, dates)
  data/reference/                     donnees de reference de Toulouse

Colonnes de chaque fichier de ville (une ligne par releve, environ une par
minute et par station) :
  fetched_at            instant du releve (datetime, UTC)
  system_id             identifiant du reseau (= nom du fichier)
  station_id            identifiant de la station
  name                  nom de la station
  lat, lon              coordonnees GPS
  capacity              nombre total d'emplacements
  num_bikes_available   velos disponibles a l'instant fetched_at
  num_docks_available   places libres pour reposer un velo
  is_renting            la station prete-t-elle des velos
  is_returning          la station accepte-t-elle les retours

Dependances : pandas et pyarrow  (pip install pandas pyarrow)
Lancement    : python -m src.lire_parquet   (ou python src/lire_parquet.py)
"""

from pathlib import Path
import pandas as pd

# Dossier des donnees, calcule a partir de l'emplacement de ce script.
RACINE = Path(__file__).resolve().parents[1]
DOSSIER_VILLES = RACINE / "data" / "status_villes"


def lister_villes():
    """Renvoie la liste des villes disponibles (un fichier Parquet par ville)."""
    fichiers = sorted(DOSSIER_VILLES.glob("*.parquet"))
    return [f.stem for f in fichiers]


def charger_ville(ville):
    """Charge le Parquet d'une ville entiere dans un DataFrame pandas."""
    chemin = DOSSIER_VILLES / f"{ville}.parquet"
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    return pd.read_parquet(chemin)


def charger_colonnes(ville, colonnes):
    """Charge seulement certaines colonnes : plus rapide et moins de memoire.

    Utile pour les grosses villes (Lyon, Toulouse) quand on n'a pas besoin de
    tout. Exemple : charger_colonnes("lyon", ["fetched_at", "station_id",
    "num_bikes_available"]).
    """
    chemin = DOSSIER_VILLES / f"{ville}.parquet"
    return pd.read_parquet(chemin, columns=colonnes)


def apercu(df):
    """Affiche un apercu simple : taille, periode, colonnes, premieres lignes."""
    print("Nombre de lignes   :", len(df))
    print("Nombre de stations :", df["station_id"].nunique())
    print("Periode            :", df["fetched_at"].min(), "a", df["fetched_at"].max())
    print("Colonnes           :", list(df.columns))
    print(df.head())


def extraire_station(df, station_id):
    """Extrait l'historique d'une seule station, trie par temps croissant."""
    une = df[df["station_id"] == station_id].copy()
    une = une.sort_values("fetched_at")
    return une


def extraire_periode(df, debut, fin):
    """Garde les releves entre deux dates (texte, par exemple 2026-05-20).

    Les horodatages sont en UTC, donc on compare en UTC.
    """
    t = pd.to_datetime(df["fetched_at"], utc=True)
    masque = (t >= pd.Timestamp(debut, tz="UTC")) & (t <= pd.Timestamp(fin, tz="UTC"))
    return df[masque]


def velos_moyens_par_station(df):
    """Pour chaque station, calcule le nombre moyen de velos disponibles."""
    resume = (
        df.groupby(["station_id", "name"])["num_bikes_available"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"num_bikes_available": "velos_moyens"})
        .sort_values("velos_moyens", ascending=False)
    )
    return resume


def main():
    # 1. Quelles villes sont disponibles
    villes = lister_villes()
    print("Villes disponibles :", len(villes))
    print(", ".join(villes))
    print()

    # 2. On charge une ville de taille moyenne pour la demonstration
    ville = "nantes" if "nantes" in villes else villes[0]
    print("Exemple avec la ville :", ville)
    print()
    df = charger_ville(ville)

    # 3. Apercu du contenu
    apercu(df)
    print()

    # 4. Statistique simple : les stations avec le plus de velos en moyenne
    resume = velos_moyens_par_station(df)
    print("Stations avec le plus de velos en moyenne :")
    print(resume.head(10).to_string(index=False))
    print()

    # 5. Historique d'une station precise
    station = df["station_id"].iloc[0]
    une = extraire_station(df, station)
    colonnes_utiles = ["fetched_at", "num_bikes_available", "num_docks_available"]
    print("Historique de la station", station, ":", len(une), "releves")
    print(une[colonnes_utiles].head())
    print()

    # 6. Export d'un extrait vers un CSV lisible
    #    On ecrit a la racine de data/, qui est ignoree par git (data/*.csv),
    #    donc cet extrait ne sera pas versionne par accident.
    sortie = RACINE / "data" / f"extrait_{ville}_{station}.csv"
    une[colonnes_utiles].to_csv(sortie, index=False)
    print("Extrait enregistre dans :", sortie)


# Pour les tres gros fichiers, on peut interroger le Parquet en SQL sans tout
# charger en memoire, avec DuckDB (pip install duckdb). Exemple :
#
#   import duckdb
#   duckdb.sql(
#       "SELECT station_id, name, avg(num_bikes_available) AS velos_moyens "
#       "FROM 'data/status_villes/lyon.parquet' "
#       "GROUP BY 1, 2 ORDER BY velos_moyens DESC LIMIT 10"
#   ).show()


if __name__ == "__main__":
    main()
