#!/usr/bin/env python
"""
Script to run the cross-model benchmarking comparison and generate charts.
Usage:
    python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PATHS, MODELS
from src.evaluation.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run cross-model evaluation benchmark.")
    parser.add_argument("--results-csv", type=str, default=None,
                        help="Path to a results_summary.csv file.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to the active run directory (to look for results_summary.csv).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save comparison outputs and figures.")
    args = parser.parse_args()

    # Determine which results CSV to use
    results_csv_path = None
    if args.results_csv:
        results_csv_path = Path(args.results_csv)
    else:
        # Check active run dir metrics
        run_dir = Path(args.run_dir) if args.run_dir else PATHS["run_dir"]
        active_csv = run_dir / "metrics" / "results_summary.csv"
        # Check exports dir
        exports_csv = PATHS["exports_dir"] / "results_summary.csv"
        
        if active_csv.exists():
            results_csv_path = active_csv
        elif exports_csv.exists():
            results_csv_path = exports_csv
        else:
            print(f"[WARN] No results_summary.csv found at either {active_csv} or {exports_csv}.")
            print("Benchmark will run with placeholders for all models.")

    # Build results_csvs dictionary
    results_csvs = {}
    if results_csv_path and results_csv_path.exists():
        print(f"Using results summary from: {results_csv_path}")
        for model_id in MODELS:
            results_csvs[model_id] = results_csv_path
    
    output_dir = Path(args.output_dir) if args.output_dir else PATHS["project_root"] / "results" / "benchmark"
    print(f"Benchmark results will be saved to: {output_dir}")

    results = run_benchmark(
        results_csvs=results_csvs if results_csvs else None,
        output_dir=output_dir
    )

    print("\n=== Benchmarking completed successfully. ===")


if __name__ == "__main__":
    main()
