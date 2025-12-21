"""Télécharge en une commande tous les jeux de données listés dans
TabSTAR (`tabstar/datasets/all_datasets.py`), sans dépendre du repo TabSTAR.

Par défaut, on télécharge tout (OpenML + Kaggle + URL) et on stocke chaque jeu
au format CSV dans `data/tabstar_raw/<source>/`. Un cache local évite de
retélécharger si le fichier existe, sauf option `--force`.

Exemples (PowerShell) :
    # Tout télécharger (attention, c'est long !) :
    python .\\scripts\\download_all_tabstar_datasets.py

    # Uniquement OpenML, limiter à 5 jeux pour tester :
    python .\\scripts\\download_all_tabstar_datasets.py --source openml --limit 5

Pré-requis : openml, kagglehub, pandas, tqdm (déjà dans requirements.txt).
Pour Kaggle, configurez vos credentials (`~/.kaggle/kaggle.json`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep
from typing import Iterable, Tuple, Type
from urllib.error import HTTPError

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanotabstar.datasets.tabstar_all_datasets import (  # noqa: E402
    OpenMLDatasetID,
    KaggleDatasetID,
    UrlDatasetID,
)


# Imports optionnels
try:
    import openml  # type: ignore
except Exception:  # pragma: no cover
    openml = None

try:
    import kagglehub  # type: ignore
except Exception:  # pragma: no cover
    kagglehub = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --- Downloaders ---

def download_openml(dataset_id, out_dir: Path, force: bool, retries: int = 5) -> str:
    if openml is None:
        return "openml non installé"
    dst = out_dir / f"{dataset_id.name}.csv"
    if dst.exists() and not force:
        return "skip"
    for _ in range(retries):
        try:
            ds = openml.datasets.get_dataset(dataset_id.value, download_data=True, download_features_meta_data=False)
            target = ds.default_target_attribute or None
            x, y, _, _ = ds.get_data(target=target)
            df = x.copy()
            if y is not None:
                target_col = target or "target"
                df[target_col] = y
            df.to_csv(dst, index=False)
            return "ok"
        except (HTTPError, Exception) as exc:
            last = str(exc)
            sleep(10)
    return f"fail: {last}"


def download_kaggle(dataset_id, out_dir: Path, force: bool) -> str:
    if kagglehub is None:
        return "kagglehub non installé/configuré"
    if not isinstance(dataset_id.value, str) or dataset_id.value.count("/") < 2:
        return "ref kaggle invalide"
    dst = out_dir / f"{dataset_id.name}.csv"
    if dst.exists() and not force:
        return "skip"
    dataset_name, file_name = dataset_id.value.rsplit("/", 1)
    try:
        dir_path = kagglehub.dataset_download(dataset_name)
        src_file = Path(dir_path) / file_name
        df = pd.read_csv(src_file)
        df.to_csv(dst, index=False)
        return "ok"
    except Exception as exc:
        return f"fail: {exc}"


def download_url(dataset_id, out_dir: Path, force: bool, retries: int = 3) -> str:
    dst = out_dir / f"{dataset_id.name}.csv"
    if dst.exists() and not force:
        return "skip"
    last = None
    for _ in range(retries):
        try:
            df = pd.read_csv(str(dataset_id.value))
            df.to_csv(dst, index=False)
            return "ok"
        except Exception as exc:
            last = exc
            sleep(5)
    return f"fail: {last}"


# --- Main runner ---

def iter_ids(source: str, OpenMLDatasetID, KaggleDatasetID, UrlDatasetID):
    if source == "openml":
        for d in OpenMLDatasetID:
            yield "openml", d
    elif source == "kaggle":
        for d in KaggleDatasetID:
            yield "kaggle", d
    elif source == "url":
        for d in UrlDatasetID:
            yield "url", d
    else:  # all
        yield from iter_ids("openml", OpenMLDatasetID, KaggleDatasetID, UrlDatasetID)
        yield from iter_ids("kaggle", OpenMLDatasetID, KaggleDatasetID, UrlDatasetID)
        yield from iter_ids("url", OpenMLDatasetID, KaggleDatasetID, UrlDatasetID)


def main():
    parser = argparse.ArgumentParser(description="Télécharge tous les datasets TabSTAR en CSV")
    parser.add_argument("--out", type=Path, default=Path("data") / "tabstar_raw", help="Dossier de sortie")
    parser.add_argument("--source", choices=["all", "openml", "kaggle", "url"], default="all", help="Filtrer la source à télécharger")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de jeux (ordre catalogue)")
    parser.add_argument("--force", action="store_true", help="Forcer le re-téléchargement même si le fichier existe")
    args = parser.parse_args()

    todo = list(iter_ids(args.source, OpenMLDatasetID, KaggleDatasetID, UrlDatasetID))
    if args.limit is not None:
        todo = todo[: args.limit]

    ensure_dir(args.out)
    stats = {"ok": 0, "skip": 0, "fail": 0}
    failures = []

    pbar = tqdm(todo, desc="Downloading", unit="ds")
    for src, ds_id in pbar:
        out_dir = args.out / src
        ensure_dir(out_dir)
        if src == "openml":
            status = download_openml(ds_id, out_dir, force=args.force)
        elif src == "kaggle":
            status = download_kaggle(ds_id, out_dir, force=args.force)
        else:
            status = download_url(ds_id, out_dir, force=args.force)
        stats.setdefault(status.split(":")[0], 0)
        stats[status.split(":")[0]] += 1
        pbar.set_postfix(status=status)
        if status.startswith("fail"):
            failures.append((ds_id.name, status))

    print("\nRésumé :", stats)
    if failures:
        print("Échecs (à relancer éventuellement avec --force) :")
        for name, msg in failures:
            print(f" - {name}: {msg}")


if __name__ == "__main__":
    main()
