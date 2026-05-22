"""
Qualitative comparison grids: original → mask → edited, side-by-side.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import TASKS


def generate_qualitative_grid(
    results_long: pd.DataFrame,
    output_dir: Path,
    max_rows_per_task: int = 4,
) -> List[Path]:
    """
    Generate side-by-side grids: Original | Mask | Edited for each task.
    Returns list of saved image paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for task in TASKS:
        task_df = results_long[results_long["task"] == task].head(max_rows_per_task)
        if task_df.empty:
            continue

        n = len(task_df)
        # Columns: one group per model (original, edited)
        models = task_df["model_id"].unique()
        n_cols = 2 * len(models) + 1  # original + (mask, edited) per model
        # Simplify: just original | edited per model
        n_cols = 1 + len(models)

        fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
        if n == 1:
            axes = np.expand_dims(axes, 0)
        if n_cols == 1:
            axes = np.expand_dims(axes, 1)

        for i, (_, row) in enumerate(task_df.iterrows()):
            # Original
            try:
                orig = Image.open(row["original_path"]).convert("RGB")
                axes[i, 0].imshow(orig)
            except Exception:
                pass
            axes[i, 0].set_title(f"Original\n{row.get('image_id', '')}", fontsize=9)
            axes[i, 0].axis("off")

            # Edited (per model)
            for j, model_id in enumerate(models):
                model_row = task_df[
                    (task_df["image_id"] == row["image_id"]) &
                    (task_df["model_id"] == model_id)
                ]
                col = 1 + j
                if len(model_row) > 0:
                    try:
                        edited = Image.open(model_row.iloc[0]["edited_path"]).convert("RGB")
                        axes[i, col].imshow(edited)
                    except Exception:
                        pass
                    success = "✓" if model_row.iloc[0].get("attr_success", False) else "✗"
                    axes[i, col].set_title(f"{model_id} {success}", fontsize=9)
                else:
                    axes[i, col].set_title(f"{model_id}\n(no result)", fontsize=9)
                axes[i, col].axis("off")

        fig.suptitle(task.replace("_", " ").title(), fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = output_dir / f"qualitative_grid_{task}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"[SAVED] {path}")

    return saved
