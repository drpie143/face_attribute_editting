#!/usr/bin/env python
"""
Script to train the attribute classifier or diffusion LoRA adapters.
Usage:
    python scripts/run_training.py --classifier --smoke
    python scripts/run_training.py --model-id sd15 --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import seed_everything, SEED, MODELS
from src.models.classifier import train_classifier
from src.models.lora_training import build_diffusers_imagefolder, run_lora_training


def main():
    parser = argparse.ArgumentParser(description="Train attribute classifier or LoRA adapters.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--classifier", action="store_true", help="Train the attribute classifier.")
    group.add_argument("--model-id", choices=list(MODELS.keys()), help="Model ID of the LoRA to train.")
    
    parser.add_argument("--smoke", action="store_true", help="Run in smoke-test mode.")
    parser.add_argument("--force", action="store_true", help="Force training and overwrite existing runs.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.classifier:
        print("=== Training Face Attribute Classifier ===")
        ckpt_path = train_classifier(smoke=args.smoke, force=args.force)
        print(f"\n=== Classifier training completed. Checkpoint: {ckpt_path} ===")
    
    elif args.model_id:
        print(f"=== Training LoRA for model: {args.model_id} ===")
        split_name = "train_smoke" if args.smoke else "train"
        
        print("Preparing Diffusers image folder...")
        data_dir = build_diffusers_imagefolder(args.model_id, split_name=split_name, force=args.force)
        
        print("Launching LoRA training script...")
        output_dir = run_lora_training(
            model_id=args.model_id,
            data_dir=data_dir,
            smoke=args.smoke,
            force=args.force
        )
        print(f"\n=== LoRA training completed. Output directory: {output_dir} ===")


if __name__ == "__main__":
    main()
