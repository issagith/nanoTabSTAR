from __future__ import annotations

"""Mini catalogue de jeux de données tabulaires.

Cette version est volontairement courte et lisible pour un usage éducatif.
Chaque entrée précise d'où provient le jeu de données et quelle colonne
est utilisée comme cible pour l'apprentissage supervisé.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Source(str, Enum):
    OPENML = "openml"
    KAGGLE = "kaggle"
    URL = "url"


@dataclass
class DatasetEntry:
    name: str
    source: Source
    ref: str | int
    target: Optional[str]
    task: str = "classification"  # "classification" ou "regression"
    description: str = ""



CATALOG: Dict[str, DatasetEntry] = {
    # ☁️ OpenML (no account required)
    "mushroom": DatasetEntry(name="mushroom", source=Source.OPENML, ref=24, target=None, description="Mushroom edible/poisonous (binary)."),
    "titanic": DatasetEntry(name="titanic", source=Source.OPENML, ref=40945, target=None, description="Titanic survival (binary)."),
    "adult_income": DatasetEntry(name="adult_income", source=Source.OPENML, ref=1590, target=None, description=">50K income (binary)."),
    "credit_german": DatasetEntry(name="credit_german", source=Source.OPENML, ref=31, target=None, description="German credit risk (binary)."),
    "pendigits": DatasetEntry(name="pendigits", source=Source.OPENML, ref=32, target=None, task="classification", description="Pen-based handwritten digits (multiclass)."),
    "waveform500": DatasetEntry(name="waveform500", source=Source.OPENML, ref=60, target=None, task="classification", description="Waveform 500 (multiclass)."),
    "wine_quality": DatasetEntry(name="wine_quality", source=Source.OPENML, ref=40498, target=None, task="classification", description="Wine quality ratings (multiclass)."),
    "california_housing": DatasetEntry(name="california_housing", source=Source.OPENML, ref=44977, target=None, task="regression", description="California house prices (regression)."),
    "concrete": DatasetEntry(name="concrete", source=Source.OPENML, ref=44959, target=None, task="regression", description="Concrete compressive strength (regression)."),
    "bank_marketing": DatasetEntry(name="bank_marketing", source=Source.OPENML, ref=1461, target=None, description="Bank marketing (binary)."),
    "breast_cancer_wisconsin": DatasetEntry(name="breast_cancer_wisconsin", source=Source.OPENML, ref=15, target=None, description="Breast cancer (binary)."),
    "magic_telescope": DatasetEntry(name="magic_telescope", source=Source.OPENML, ref=44125, target=None, description="MAGIC telescope particle classification (binary)."),
    "particle_higgs": DatasetEntry(name="particle_higgs", source=Source.OPENML, ref=42769, target=None, description="HIGGS particle dataset (binary)."),
    "qsar_tid_11": DatasetEntry(name="qsar_tid_11", source=Source.OPENML, ref=3050, target=None, description="QSAR TID 11 (regression/classification depending on preprocessing)."),
    "yeast_protein": DatasetEntry(name="yeast_protein", source=Source.OPENML, ref=181, target=None, task="classification", description="Yeast protein localization (multiclass)."),
    "cover_type": DatasetEntry(name="cover_type", source=Source.OPENML, ref=1596, target=None, task="classification", description="Forest cover type (multiclass)."),
    "ozone_level": DatasetEntry(name="ozone_level", source=Source.OPENML, ref=1487, target=None, task="regression", description="Ozone level prediction (regression)."),
    "mnist_digits": DatasetEntry(name="mnist_digits", source=Source.OPENML, ref=554, target=None, task="classification", description="MNIST digits (multiclass)."),
    "bank_personal_loan": DatasetEntry(name="bank_personal_loan", source=Source.OPENML, ref=43826, target=None, description="Bank personal loan modeling (binary)."),
    "home_credit_default": DatasetEntry(name="home_credit_default", source=Source.OPENML, ref=45567, target=None, description="Home credit default risk (binary)."),
    "phishing_urls": DatasetEntry(name="phishing_urls", source=Source.OPENML, ref=43622, target=None, description="Phishing website detection (binary)."),
    "abalone": DatasetEntry(name="abalone", source=Source.OPENML, ref=42726, target=None, task="regression", description="Abalone age prediction (regression)."),
    "diabetes_sklearn": DatasetEntry(name="diabetes_sklearn", source=Source.OPENML, ref=44223, target=None, task="regression", description="Diabetes regression dataset."),
    "sepsis_physionet": DatasetEntry(name="sepsis_physionet", source=Source.OPENML, ref=46677, target=None, description="Sepsis detection (healthcare)."),
    "satellite": DatasetEntry(name="satellite", source=Source.OPENML, ref=40900, target=None, description="Satellite image classification (multiclass)."),
    "pc4_defects": DatasetEntry(name="pc4_defects", source=Source.OPENML, ref=1049, target=None, description="PC4 code defects (binary)."),
    "kc1_defects": DatasetEntry(name="kc1_defects", source=Source.OPENML, ref=1067, target=None, description="KC1 code defects (binary)."),
    "skin_segmentation": DatasetEntry(name="skin_segmentation", source=Source.OPENML, ref=1502, target=None, description="Skin segmentation (multiclass/binary)."),
    "opt_digits": DatasetEntry(name="opt_digits", source=Source.OPENML, ref=28, target=None, task="classification", description="Optical recognition of handwritten digits (multiclass)."),
    "letter_recognition": DatasetEntry(name="letter_recognition", source=Source.OPENML, ref=6, target=None, task="classification", description="Letter recognition (multiclass)."),
    "page_block": DatasetEntry(name="page_block", source=Source.OPENML, ref=30, target=None, task="classification", description="Page block classification (multiclass)."),
    "isolet": DatasetEntry(name="isolet", source=Source.OPENML, ref=300, target=None, task="classification", description="ISOLET spoken letter recognition (multiclass)."),
    "spambase": DatasetEntry(name="spambase", source=Source.OPENML, ref=44, target=None, description="Spam email detection (binary)."),
    "mushroom_large": DatasetEntry(name="mushroom_large", source=Source.OPENML, ref=41158, target=None, description="Alternate mushroom anonymized dataset (binary)."),
    "kddcup98_direct_mail": DatasetEntry(name="kddcup98_direct_mail", source=Source.OPENML, ref=42343, target=None, description="KDD Cup 98 direct mail (binary)."),
    "twitter_disaster": DatasetEntry(name="twitter_disaster", source=Source.OPENML, ref=43395, target=None, description="Twitter disaster detection (binary)."),
    "fish_toxicity": DatasetEntry(name="fish_toxicity", source=Source.OPENML, ref=44970, target=None, task="regression", description="Fish toxicity (regression)."),
    "superconductivity": DatasetEntry(name="superconductivity", source=Source.OPENML, ref=44964, target=None, task="regression", description="Superconductivity regression dataset."),
    "cifar10": DatasetEntry(name="cifar10", source=Source.OPENML, ref=40927, target=None, task="classification", description="CIFAR-10 (image classification, multiclass)."),
    "us_congress_votes": DatasetEntry(name="us_congress_votes", source=Source.OPENML, ref=56, target=None, task="classification", description="US Congress votes (binary/multiclass)."),
    # 🌐 Kaggle examples (require Kaggle credentials for full download)
    # "kaggle_yelp_reviews": DatasetEntry(name="kaggle_yelp_reviews", source=Source.KAGGLE, ref="omkarsabnis/yelp-reviews-dataset/yelp.csv", target=None, task="classification", description="Yelp reviews (textual sentiment)."),
    # "kaggle_michelin": DatasetEntry(name="kaggle_michelin", source=Source.KAGGLE, ref="ngshiheng/michelin-guide-restaurants-2021/michelin_my_maps.csv", target=None, description="Michelin restaurants dataset (categorical/text)."),
    # "kaggle_used_cars_pakistan": DatasetEntry(name="kaggle_used_cars_pakistan", source=Source.KAGGLE, ref="mustafaimam/used-car-prices-in-pakistan-2021/Used_car_prices_in_Pakistan_cleaned.csv", target=None, task="regression", description="Used car prices (regression)."),
    # "kaggle_movies_metadata": DatasetEntry(name="kaggle_movies_metadata", source=Source.KAGGLE, ref="rounakbanik/the-movies-dataset/movies_metadata.csv", target=None, description="Movies metadata (textual + numeric)."),
    # "kaggle_spotify": DatasetEntry(name="kaggle_spotify", source=Source.KAGGLE, ref="maharshipandya/-spotify-tracks-dataset/dataset.csv", target=None, description="Spotify tracks popularity dataset (regression/classification)."),
    # "kaggle_museums": DatasetEntry(name="kaggle_museums", source=Source.KAGGLE, ref="markusschmitz/museums/museums_prep.csv", target=None, description="US museums revenues (regression)."),
    # 🌐 URL-hosted small datasets
    "goodreads_books": DatasetEntry(name="goodreads_books", source=Source.URL, ref="http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv", target=None, description="Goodreads books dataset (textual)."),
    "rotten_tomatoes": DatasetEntry(name="rotten_tomatoes", source=Source.URL, ref="http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv", target=None, description="Rotten Tomatoes small dataset (text + numeric)."),
    "scimagojr": DatasetEntry(name="scimagojr", source=Source.URL, ref="https://www.scimagojr.com/journalrank.php?out=xls", target=None, description="SCImago journal rankings (tabular)."),
    "baby_products": DatasetEntry(name="baby_products", source=Source.URL, ref="http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv", target=None, description="Babies R Us products (regression/pricing)."),
    "bike_prices": DatasetEntry(name="bike_prices", source=Source.URL, ref="http://pages.cs.wisc.edu/~anhai/data/784_data/bikes/csv_files/bikewale.csv", target=None, description="Bike pricing dataset (regression)."),
}

DEFAULT_DATASETS = list(CATALOG.keys())


def list_datasets() -> Dict[str, str]:
    """Retourne nom -> description pour affichage rapide."""
    return {name: entry.description for name, entry in CATALOG.items()}
