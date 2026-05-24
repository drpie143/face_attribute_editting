"""
Generate the final Markdown report summarizing the benchmark.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from src.config import (
    EDIT_CONFIG,
    MODELS,
    PATHS,
    RUN_TAG,
    TASKS,
    load_jsonl,
    train_steps_for_model,
)


def generate_report(
    results_summary: Optional[pd.DataFrame] = None,
    qual_grid_paths: Optional[List[Path]] = None,
    run_dir: Optional[Path] = None,
    models_list: Optional[List[str]] = None,
    smoke: bool = False,
) -> Path:
    """Generate the final benchmark report as Markdown."""
    run_dir = Path(run_dir or PATHS["run_dir"])
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "final_report.md"
    models_list = models_list or list(MODELS.keys())

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    lines = [
        "# Face Attribute Editing Benchmark Report",
        "",
        f"- Run tag: `{RUN_TAG}`",
        f"- Smoke test: `{smoke}`",
        f"- Edit engine: `{EDIT_CONFIG['engine']}` for all models",
        f"- Created: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## 1. Goal",
        "",
        "Benchmark SD1.5 and SDXL with the same mask-native inpainting "
        "workflow for face attribute editing.",
        "",
        "## 2. Environment",
        "",
    ]

    for key, value in env.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.extend([
        "## 3. Dataset and split",
        "",
    ])
    manifest_dir = PATHS["manifest_dir"]
    for name in ["train", "val", "test", "train_smoke", "val_smoke", "test_smoke"]:
        path = manifest_dir / f"{name}.jsonl"
        count = len(load_jsonl(path)) if path.exists() else 0
        lines.append(f"- {name}: {count}")
    lines.append("")

    lines.extend([
        "## 4. Models",
        "",
    ])
    for model_id in models_list:
        cfg = MODELS[model_id]
        lines.append(
            f"- `{model_id}`: `{cfg['pretrained_model_name_or_path']}`, "
            f"arch=`{cfg['architecture']}`, resolution={cfg['resolution']}, "
            f"rank={cfg['rank']}, steps={train_steps_for_model(model_id, smoke)}"
        )
    lines.append("")

    lines.extend([
        "## 5. Quantitative results",
        "",
    ])
    if results_summary is not None and len(results_summary) > 0:
        lines.append(results_summary.to_markdown(index=False))
    else:
        lines.append("_No evaluation results yet._")
    lines.append("")

    lines.extend([
        "## 6. Qualitative grids",
        "",
    ])
    if qual_grid_paths:
        for path in qual_grid_paths:
            try:
                rel = path.relative_to(run_dir)
            except ValueError:
                rel = path
            lines.append(f"- `{rel}`")
    else:
        lines.append("_No qualitative grids generated yet._")
    lines.append("")

    lines.extend([
        "## 7. Known failure modes",
        "",
        "- Identity drift, especially for `make_older` with high strength.",
        "- Distorted eyeglasses or asymmetric frames.",
        "- Distorted teeth or mouth shape for `make_smiling`.",
        "- Hair or background changes when the age mask is too broad.",
        "",
        "## 8. Limitations",
        "",
        "- LoRA training uses Diffusers text-to-image LoRA examples; inpainting "
        "is applied at inference with masks.",
        "- Identity preservation is inspected qualitatively in this report; the "
        "public benchmark focuses on attribute and background-preservation metrics.",
        "- Playground v2.5 is excluded from the final report because visual "
        "inspection showed noisy inpainting artifacts.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")
    return report_path
