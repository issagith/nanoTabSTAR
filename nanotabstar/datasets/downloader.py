from __future__ import annotations

"""Téléchargement léger inspiré de TabSTAR.

Fonctionne avec OpenML (par défaut), Kaggle et des URLs simples.
On retourne un DataFrame prêt à être pré-traité + le nom de la colonne cible.
"""
import os
from time import sleep
from urllib.error import HTTPError
from typing import Tuple

import pandas as pd

from .catalog import CATALOG, DatasetEntry, Source

# Import paresseux pour ne pas obliger Kaggle/OpenML si on ne les utilise pas.
try:
    import openml  # type: ignore
except Exception:  # pragma: no cover - dépendance optionnelle
    openml = None

try:
    import kagglehub  # type: ignore
except Exception:  # pragma: no cover - dépendance optionnelle
    kagglehub = None


def fetch_dataset(
    name: str,
    cache_dir: str = "data/raw",
    max_rows: int | None = None,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Récupère un dataset par son nom catalogue.

    - Si un CSV est déjà en cache, on le relit (sauf force_download).
    - `max_rows` permet de réduire la taille pour des runs rapides/éducatifs.
    - Retourne (dataframe, target_col).
    """
    if name not in CATALOG:
        raise KeyError(f"Dataset '{name}' non trouvé. Disponibles: {list(CATALOG)}")

    entry = CATALOG[name]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _find_cached_path(entry, cache_dir)

    if cache_path and os.path.exists(cache_path) and not force_download:
        df = pd.read_csv(cache_path)
        target_col = _resolve_target(entry, df, fallback_allowed=True)
        df = _maybe_subsample(df, max_rows)
        return df, target_col

    if entry.source == Source.OPENML:
        df, target_col = _load_openml(entry)
    elif entry.source == Source.KAGGLE:
        df, target_col = _load_kaggle(entry)
    elif entry.source == Source.URL:
        df, target_col = _load_url(entry)
    else:
        raise ValueError(f"Source inconnue: {entry.source}")

    df = _maybe_subsample(df, max_rows)
    # Enregistre sous le nom canonique du catalogue pour les prochains runs
    canonical_path = os.path.join(cache_dir, f"{entry.name}.csv")
    df.to_csv(canonical_path, index=False)
    return df, target_col


# --- Loaders ---

def _load_openml(entry: DatasetEntry):
    if openml is None:
        raise ImportError("openml n'est pas installé. Ajoutez-le via requirements.txt")

    target_col = None
    for _ in range(10):
        try:
            print(f"💾 Téléchargement OpenML [{entry.name}] (id={entry.ref})")
            dataset = openml.datasets.get_dataset(
                entry.ref,
                download_data=True,
                download_features_meta_data=False,
            )
            target_col = entry.target or dataset.default_target_attribute
            x, y, _, _ = dataset.get_data(target=target_col)
            df = x.copy()
            df[target_col] = y
            return df, target_col
        except (Exception, HTTPError) as e:  # OpenML lève plusieurs exceptions custom
            print(f"⚠️ OpenML error: {e}. Retry in 30s...")
            sleep(30)
    raise RuntimeError("Echec du téléchargement OpenML après plusieurs tentatives")


def _load_kaggle(entry: DatasetEntry):
    if kagglehub is None:
        raise ImportError("kagglehub n'est pas installé ou mal configuré")
    if not isinstance(entry.ref, str) or entry.ref.count('/') < 2:
        raise ValueError("Pour Kaggle, 'ref' doit être 'namespace/dataset/file.csv'")
    dataset_name, file = entry.ref.rsplit('/', 1)
    print(f"💾 Téléchargement Kaggle [{entry.name}] depuis {dataset_name} -> {file}")
    dir_path = kagglehub.dataset_download(dataset_name)
    df = pd.read_csv(os.path.join(dir_path, file))
    target_col = _resolve_target(entry, df)
    return df, target_col


def _load_url(entry: DatasetEntry):
    if not isinstance(entry.ref, str):
        raise ValueError("Pour URL, 'ref' doit être une string")
    for _ in range(5):
        try:
            print(f"💾 Téléchargement URL [{entry.name}] depuis {entry.ref}")
            df = pd.read_csv(entry.ref)
            target_col = _resolve_target(entry, df)
            return df, target_col
        except HTTPError as e:
            print(f"⚠️ HTTP error: {e}. Retry in 30s...")
            sleep(30)
    raise RuntimeError("Echec du téléchargement URL après plusieurs tentatives")


# --- Helpers ---

def _maybe_subsample(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def _resolve_target(entry: DatasetEntry, df: pd.DataFrame, fallback_allowed: bool = False) -> str:
    """Détermine la colonne cible.

    - si entry.target est défini, on l'utilise
    - sinon, on essaie de deviner : dernière colonne du cache si fallback_allowed
    """
    if entry.target:
        return entry.target
    if fallback_allowed:
        return df.columns[-1]
    raise ValueError("Impossible de déterminer la colonne cible; précisez entry.target")


def _find_cached_path(entry: DatasetEntry, cache_dir: str) -> str | None:
    """Essaie de retrouver un CSV déjà téléchargé avec divers noms.

    On cherche, dans l'ordre :
    - cache_dir/<entry.name>.csv
    - cache_dir/<entry.ref>.csv (si ref est un int)
    - cache_dir/<basename-of-ref>.csv (pour Kaggle/URL)
    - sous-dossiers typiques: openml/, kaggle/, url/ avec les mêmes variantes
    - enfin, on scanne cache_dir pour un fichier CSV dont le nom contient ref ou entry.name
    """
    candidates = []

    def add(path):
        candidates.append(os.path.join(cache_dir, path))

    add(f"{entry.name}.csv")
    if isinstance(entry.ref, int):
        add(f"{entry.ref}.csv")
    else:
        basename = os.path.basename(str(entry.ref))
        if basename:
            add(basename)
    for sub in ("openml", "kaggle", "url"):
        base = os.path.join(sub, f"{entry.name}.csv")
        add(base)
        if isinstance(entry.ref, int):
            add(os.path.join(sub, f"{entry.ref}.csv"))
        else:
            basename = os.path.basename(str(entry.ref))
            if basename:
                add(os.path.join(sub, basename))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Fallback: scan cache_dir for any CSV containing name or ref
    ref_str = str(entry.ref)
    for root, _, files in os.walk(cache_dir):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            stem = f.lower()
            if entry.name.lower() in stem or ref_str in stem:
                return os.path.join(root, f)
    return None
