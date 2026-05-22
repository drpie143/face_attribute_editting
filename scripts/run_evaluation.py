#!/usr/bin/env python
"""
Script to run evaluation on edited image outputs.
Usage:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PATHS
from src.evaluation.evaluate import evaluate_edits


def main():
    parser = argparse.ArgumentParser(description="Evaluate edited outputs.")
    parser.add_argument("--run-dir", type=str, default=None, help="Project run directory.")
    parser.add_argument("--classifier-path", type=str, default=None, help="Path to classifier best.pt.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else PATHS["run_dir"]
    classifier_path = Path(args.classifier_path) if args.classifier_path else None

    print(f"=== Evaluating Edited Outputs in {run_dir} ===")
    results_long, results_summary = evaluate_edits(
        run_dir=run_dir,
        classifier_path=classifier_path
    )
    
    if len(results_long) == 0:
        print("\n=== Evaluation finished with warnings: No edited meta files found. ===")
    else:
        print(f"\n=== Evaluation completed successfully. Summarized {len(results_summary)} rows. ===")
        
        # 1. Generate qualitative side-by-side comparison grids
        print("\n=== Generating Qualitative Grids ===")
        from src.visualization.qualitative_grid import generate_qualitative_grid
        qual_dir = run_dir / "reports" / "qualitative"
        qual_grid_paths = generate_qualitative_grid(
            results_long=results_long,
            output_dir=qual_dir,
            max_rows_per_task=4
        )
        print(f"Generated {len(qual_grid_paths)} qualitative grid images in: {qual_dir}")
        
        # 2. Compile final benchmark report
        print("\n=== Compiling Benchmark Report ===")
        from src.visualization.report import generate_report
        report_path = generate_report(
            results_summary=results_summary,
            qual_grid_paths=qual_grid_paths,
            run_dir=run_dir,
            smoke=False
        )
        print(f"Final report saved to: {report_path}")


if __name__ == "__main__":
    main()

