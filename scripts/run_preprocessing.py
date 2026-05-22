#!/usr/bin/env python
"""
Script to run data preprocessing and split/manifest building.
Usage:
    python scripts/run_preprocessing.py --max-images 2000 --force
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import seed_everything, SEED
from src.data.preprocess import run_preprocessing
from src.data.manifest import build_manifests


def main():
    parser = argparse.ArgumentParser(description="Preprocess CelebAMask-HQ data and build manifests.")
    parser.add_argument("--max-images", type=int, default=2000, help="Maximum images to process.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing processed data.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for split reproducibility.")
    args = parser.parse_args()

    seed_everything(args.seed)

    print("=== Step 1: Preprocessing raw dataset and building masks ===")
    records = run_preprocessing(max_images=args.max_images, force=args.force)

    print("\n=== Step 2: Creating train/val/test splits and saving manifests ===")
    splits = build_manifests(records, seed=args.seed, force=args.force)
    
    print("\n=== Preprocessing successfully completed ===")


if __name__ == "__main__":
    main()
