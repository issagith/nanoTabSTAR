"""Construit un dump HDF5 à partir du catalogue nanoTabStar.

Exemple rapide (Windows PowerShell):
python .\\scripts\\create_dump.py --max-rows 500
"""
import argparse
import os
import sys

# Add parent dir to path to import nanotabstar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanotabstar.datasets import build_h5_corpus, list_datasets, DEFAULT_DATASETS


def main():
    parser = argparse.ArgumentParser(description="Télécharge et prépare les datasets nanoTabStar")
    parser.add_argument("--output", default="data/pretrain_corpus.h5", help="Chemin du fichier HDF5 de sortie")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Noms des datasets à inclure. Omettre ou utiliser --all pour tout prendre.",
    )
    parser.add_argument("--all", action="store_true", help="Inclure tout le catalogue sans le lister manuellement")
    parser.add_argument("--max-rows", type=int, default=5000, help="Sous-échantillonnage max par dataset")
    parser.add_argument("--force-download", action="store_true", help="Ignore le cache CSV local")
    parser.add_argument("--cache-dir", default="data/raw", help="Répertoire cache CSV (utilisez data/tabstar_raw pour réutiliser download_all)")
    parser.add_argument("--tokenizer", default="intfloat/e5-small-v2", help="Backbone textuel pour le verbaliseur")
    parser.add_argument("--max-token-len", type=int, default=32, help="Longueur maximale des séquences tokenisées")
    args = parser.parse_args()

    available = list_datasets()
    print("Datasets disponibles :")
    for name, desc in available.items():
        print(f" - {name}: {desc}")

    if args.all or not args.datasets:
        selected = DEFAULT_DATASETS
    else:
        selected = args.datasets

    print(f"\nSélection: {len(selected)} datasets")

    written = build_h5_corpus(
        h5_path=args.output,
        datasets=selected,
        max_rows=args.max_rows,
        force_download=args.force_download,
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer,
        max_token_len=args.max_token_len,
    )
    print(f"\n✅ Dump créé: {args.output} -> {written}")


if __name__ == "__main__":
    main()