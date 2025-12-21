from __future__ import annotations

"""Pipeline éducative : téléchargement -> prétraitement -> dump HDF5.

Usage rapide
------------
>>> from nanotabstar.datasets.builder import build_h5_corpus
>>> build_h5_corpus(max_rows=500)

Cela crée `data/pretrain_corpus.h5` en réutilisant le préprocesseur
TabSTAR existant, avec un sous-échantillonnage optionnel (`max_rows`).
"""
import os
from typing import Iterable, List

import h5py
from tqdm import tqdm

from nanotabstar.preparation import TabStarPreprocessor
from .catalog import CATALOG, DEFAULT_DATASETS, DatasetEntry
from .downloader import fetch_dataset


def save_to_h5(processed_data: dict, h5_file: h5py.File, dataset_name: str) -> None:
    """Enregistre les tenseurs procesés dans un groupe HDF5."""
    grp = h5_file.create_group(dataset_name)
    grp.create_dataset("feature_input_ids", data=processed_data["feature_input_ids"], compression="gzip")
    grp.create_dataset("feature_num_values", data=processed_data["feature_num_values"], compression="gzip")
    grp.create_dataset("target_input_ids", data=processed_data["target_input_ids"])
    grp.create_dataset("labels", data=processed_data["labels"])
    grp.attrs["n_classes"] = processed_data.get("n_classes", 0)


def build_h5_corpus(
    h5_path: str = "data/pretrain_corpus.h5",
    datasets: Iterable[str] | None = None,
    max_rows: int | None = None,
    force_download: bool = False,
    cache_dir: str = "data/raw",
    tokenizer_name: str = "intfloat/e5-small-v2",
    max_token_len: int = 32,
) -> List[str]:
    """Construit un dump HDF5 multi-datasets.

    Args:
        h5_path: chemin de sortie
        datasets: noms du catalogue à inclure. Par défaut, tous les datasets de `catalog.py`.
        max_rows: limite de lignes par dataset (sous-échantillonnage aléatoire reproductible).
        force_download: ignore le cache CSV si True
        cache_dir: répertoire cache CSV (permet de réutiliser les dumps déjà téléchargés,
            ex: ceux créés par download_all_tabstar_datasets.py)
        tokenizer_name/max_token_len: paramètres du verbaliseur TabSTAR

    Returns:
        La liste des datasets effectivement écrits.
    """
    selected = list(datasets) if datasets else DEFAULT_DATASETS
    for ds in selected:
        if ds not in CATALOG:
            raise KeyError(f"Dataset '{ds}' n'existe pas dans le catalogue. Disponibles: {list(CATALOG)}")

    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    preprocessor = TabStarPreprocessor(model_name=tokenizer_name, max_token_len=max_token_len)

    written: List[str] = []
    with h5py.File(h5_path, "w") as h5_file:
        for name in tqdm(selected, desc="Building H5 dump"):
            entry: DatasetEntry = CATALOG[name]
            df, target_col = fetch_dataset(name, cache_dir=cache_dir, max_rows=max_rows, force_download=force_download)
            processed = preprocessor.process_dataset(df, target_col=target_col, task_type=entry.task)
            save_to_h5(processed, h5_file, dataset_name=name)
            written.append(name)
    return written
