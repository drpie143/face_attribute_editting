"""
Generate the final Markdown report summarising the benchmark.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """
    Generate the final benchmark report as Markdown.
    Returns the path to the saved report file.
    """
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
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
    }

    lines = []
    lines.append("# Face Attribute Editing Benchmark — Report")
    lines.append("")
    lines.append(f"- Run tag: `{RUN_TAG}`")
    lines.append(f"- Smoke test: `{smoke}`")
    lines.append(f"- Edit engine: `{EDIT_CONFIG['engine']}` for all models")
    lines.append(f"- Created: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")

    lines.append("## 1. Goal")
    lines.append("")
    lines.append("Benchmark SD1.5, SDXL, and Playground v2.5 with the same "
                 "mask-native inpainting workflow for face attribute editing.")
    lines.append("")

    lines.append("## 2. Environment")
    lines.append("")
    for k, v in env.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## 3. Dataset and split")
    lines.append("")
    manifest_dir = PATHS["manifest_dir"]
    for name in ["train", "val", "test", "train_smoke", "val_smoke", "test_smoke"]:
        p = manifest_dir / f"{name}.jsonl"
        n = len(load_jsonl(p)) if p.exists() else 0
        lines.append(f"- {name}: {n}")
    lines.append("")

    lines.append("## 4. Models")
    lines.append("")
    for model_id in models_list:
        cfg = MODELS[model_id]
        lines.append(
            f"- `{model_id}`: `{cfg['pretrained_model_name_or_path']}`, "
            f"arch=`{cfg['architecture']}`, resolution={cfg['resolution']}, "
            f"rank={cfg['rank']}, steps={train_steps_for_model(model_id, smoke)}"
        )
    lines.append("")

    lines.append("## 5. Quantitative results")
    lines.append("")
    if results_summary is not None and len(results_summary) > 0:
        lines.append(results_summary.to_markdown(index=False))
    else:
        lines.append("_No evaluation results yet._")
    lines.append("")

    lines.append("## 6. Qualitative grids")
    lines.append("")
    if qual_grid_paths:
        for p in qual_grid_paths:
            try:
                rel = p.relative_to(run_dir)
            except ValueError:
                rel = p
            lines.append(f"- `{rel}`")
    else:
        lines.append("_No qualitative grids generated yet._")
    lines.append("")

    lines.append("## 7. Known failure modes")
    lines.append("")
    lines.append("- Identity drift, especially for `make_older` with high strength.")
    lines.append("- Distorted eyeglasses or asymmetric frames.")
    lines.append("- Distorted teeth/mouth for `make_smiling`.")
    lines.append("- Hair/background changes when age mask is too broad.")
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("")
    lines.append("- LoRA training uses Diffusers text-to-image LoRA examples; "
                 "inpainting is applied at inference with hard masks.")
    lines.append("- InsightFace identity is optional; if unavailable, "
                 "identity fields may be empty.")
    lines.append("- Playground v2.5 results may be pending.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")
    return report_path
