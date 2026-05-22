#!/usr/bin/env python
"""
Script to run batch inference/attribute editing on the test set.
Usage:
    python scripts/run_inference.py --model-id sd15 --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import seed_everything, SEED, MODELS
from src.inference.batch_edit import batch_edit_model


def main():
    parser = argparse.ArgumentParser(description="Run batch face attribute editing inference.")
    parser.add_argument("--model-id", choices=list(MODELS.keys()), required=True, help="Model ID of the LoRA to train.")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke-test mode (2 samples per task).")
    parser.add_argument("--samples-per-task", type=int, default=None, help="Number of samples to edit per task.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    seed_everything(args.seed)

    split_name = "test_smoke" if args.smoke else "test"
    
    # If samples_per_task is not specified, default to 2 for smoke test and 50 for full test
    if args.samples_per_task is None:
        samples_per_task = 2 if args.smoke else 50
    else:
        samples_per_task = args.samples_per_task

    print(f"=== Running Batch Inference for Model: {args.model_id} ===")
    print(f"Split: {split_name}, Samples per task: {samples_per_task}")
    
    metadata = batch_edit_model(
        model_id=args.model_id,
        split_name=split_name,
        samples_per_task=samples_per_task
    )
    
    print(f"\n=== Inference completed. Edited {len(metadata)} images ===")


if __name__ == "__main__":
    main()
