"""Sous-module datasets de nanoTabStar.

Objectifs :
- Centraliser un petit catalogue de jeux de données tabulaires.
- Télécharger (OpenML/Kaggle/URL) et mettre en cache au format CSV.
- Convertir en dump HDF5 prêt pour le prétraining TabSTAR.
"""
from .catalog import CATALOG, DEFAULT_DATASETS, DatasetEntry, Source, list_datasets
from .builder import build_h5_corpus
from .downloader import fetch_dataset

__all__ = [
    "CATALOG",
    "DEFAULT_DATASETS",
    "DatasetEntry",
    "Source",
    "list_datasets",
    "fetch_dataset",
    "build_h5_corpus",
]
