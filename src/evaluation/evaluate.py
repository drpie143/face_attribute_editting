"""
Evaluate edited images: compute attribute accuracy, LPIPS, SSIM, etc.
Produces results_long.csv (per-image) and results_summary.csv (per model/task).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.config import (
    EDIT_CONFIG,
    PATHS,
    TASK_TO_ATTR_INDEX,
    CLASSIFIER_CONFIG,
    ATTR_COLUMNS,
)
from src.evaluation.metrics import (
    background_l1,
    background_ssim,
    identity_similarity,
    lpips_distance,
)
from src.models.classifier import build_attr_classifier, predict_attrs


def load_classifier_for_eval(path: Path):
    """Load trained classifier for evaluation. Returns (model, device, thresholds)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {path}. "
            "Train one with `python scripts/run_training.py --classifier` "
            "or pass --classifier-path."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_attr_classifier(False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    thresholds = ckpt.get("thresholds",
                          [CLASSIFIER_CONFIG["threshold"]] * len(ATTR_COLUMNS))
    return model, device, thresholds


def _resolve_path(stored_path: str, meta_path: Path) -> Path:
    p = Path(stored_path)
    # If the path exists as-is, return it
    if p.exists():
        return p
    # If it is inside the edited directory (like best_path or candidates)
    if "edited" in p.parts:
        local_p = meta_path.parent / p.name
        if local_p.exists():
            return local_p
    # If it is inside processed/ (like source_image or masks)
    if "processed" in p.parts:
        idx = p.parts.index("processed")
        rel_parts = p.parts[idx+1:]
        local_p = PATHS["processed_dir"] / Path(*rel_parts)
        if local_p.exists():
            return local_p
    # Fallback to name in meta_path.parent
    local_p = meta_path.parent / p.name
    if local_p.exists():
        return local_p
    return p


def evaluate_edits(run_dir: Optional[Path] = None,
                   classifier_path: Optional[Path] = None) -> tuple:
    """
    Evaluate all edited images in run_dir/edited/.

    Returns:
        (results_long: pd.DataFrame, results_summary: pd.DataFrame)
    """
    run_dir = Path(run_dir or PATHS["run_dir"])
    if classifier_path is None:
        classifier_path = run_dir / "checkpoints" / "attr_classifier" / "best.pt"

    meta_files = sorted((run_dir / "edited").glob("*/*/*_meta.json"))
    if not meta_files:
        print("[WARN] No edited meta files found.")
        return pd.DataFrame(), pd.DataFrame()

    model, device, thresholds = load_classifier_for_eval(classifier_path)
    rows = []

    for meta_path in tqdm(meta_files, desc="evaluate"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task = meta["task"]
        idx = TASK_TO_ATTR_INDEX[task]

        orig_path = _resolve_path(meta["source_image"], meta_path)
        edited_path = _resolve_path(meta["best_path"], meta_path)
        mask_path = _resolve_path(meta.get("mask_soft", meta.get("mask_hard")), meta_path)

        orig = Image.open(orig_path).convert("RGB")
        edited = Image.open(edited_path).convert("RGB").resize(
            orig.size, Image.Resampling.LANCZOS)
        mask = Image.open(mask_path).convert("L").resize(
            orig.size, Image.Resampling.BILINEAR)

        p_before = predict_attrs(model, device, orig)
        p_after = predict_attrs(model, device, edited)

        if task == "make_older":
            target = 0.0
            attr_success = p_after[idx] < thresholds[idx]
            attr_score = 1.0 - float(p_after[idx])
            attr_delta = max(0.0, float(p_before[idx]) - float(p_after[idx]))
        else:
            target = 1.0
            attr_success = p_after[idx] >= thresholds[idx]
            attr_score = float(p_after[idx])
            attr_delta = max(0.0, float(p_after[idx]) - float(p_before[idx]))

        selected = meta["candidates"][meta["selected_candidate"]]
        ident = selected.get("identity_similarity")
        if ident is None:
            ident = identity_similarity(orig, edited)

        rows.append({
            "model_id": meta["model_id"],
            "engine": meta.get("engine", "inpainting"),
            "task": task,
            "image_id": meta["id"],
            "candidate_id": meta["selected_candidate"],
            "selection_score": meta.get("selection_score",
                                         selected.get("selection_score")),
            "attr_target": target,
            "attr_prob_before": float(p_before[idx]),
            "attr_prob_after": float(p_after[idx]),
            "attr_score": attr_score,
            "attr_delta": attr_delta,
            "attr_direction_success": bool(attr_delta >= 0.05),
            "attr_success": bool(attr_success),
            "identity_similarity": ident,
            "lpips": lpips_distance(orig, edited),
            "background_l1": background_l1(orig, edited, mask),
            "background_ssim": background_ssim(orig, edited, mask),
            "strength": meta["strength"],
            "guidance_scale": meta["guidance_scale"],
            "seed": selected["seed"],
            "original_path": str(orig_path),
            "edited_path": str(edited_path),
        })


    results_long = pd.DataFrame(rows)

    # Save per-image results
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    long_path = metrics_dir / "results_long.csv"
    results_long.to_csv(long_path, index=False)
    print(f"[SAVED] {long_path} ({len(results_long)} rows)")

    # Summary per model/task
    if len(results_long) > 0:
        agg = {
            "n": ("image_id", "count"),
            "attr_success_rate": ("attr_success", "mean"),
            "attr_direction_success_rate": ("attr_direction_success", "mean"),
            "attr_prob_after_mean": ("attr_prob_after", "mean"),
            "attr_score_mean": ("attr_score", "mean"),
            "attr_delta_mean": ("attr_delta", "mean"),
            "identity_similarity_mean": ("identity_similarity", "mean"),
            "lpips_mean": ("lpips", "mean"),
            "background_l1_mean": ("background_l1", "mean"),
            "background_ssim_mean": ("background_ssim", "mean"),
            "selection_score_mean": ("selection_score", "mean"),
        }
        results_summary = results_long.groupby(
            ["model_id", "engine", "task"], as_index=False
        ).agg(**agg)
    else:
        results_summary = pd.DataFrame()

    summary_path = metrics_dir / "results_summary.csv"
    results_summary.to_csv(summary_path, index=False)
    print(f"[SAVED] {summary_path}")

    return results_long, results_summary
