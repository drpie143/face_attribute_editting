#!/usr/bin/env python
"""
Rerank existing candidate images with attribute-focused weights.

This is useful when inference already produced multiple candidate blends but
the original selection score favored preservation too strongly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PATHS, TASKS, TASK_TO_ATTR_INDEX
from src.evaluation.evaluate import _resolve_path, load_classifier_for_eval
from src.models.classifier import predict_attrs


DEFAULT_WEIGHTS = {
    "attr_score": 0.70,
    "attr_delta": 0.15,
    "background_score": 0.10,
    "lpips_score": 0.05,
}


def parse_weights(raw: str | None) -> dict[str, float]:
    if not raw:
        return DEFAULT_WEIGHTS.copy()

    weights = DEFAULT_WEIGHTS.copy()
    for item in raw.split(","):
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value)

    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def candidate_path(candidate: dict[str, Any], meta_path: Path) -> Path:
    stored = (
        candidate.get("blend_path")
        or candidate.get("path")
        or candidate.get("raw_path")
    )
    if not stored:
        raise KeyError(f"Candidate in {meta_path} has no path/blend_path/raw_path")
    return _resolve_path(stored, meta_path)


def score_candidate(
    *,
    task: str,
    p_before: float,
    p_after: float,
    threshold: float,
    background_l1: float | None,
    lpips: float | None,
    weights: dict[str, float],
) -> dict[str, float | bool]:
    if task == "make_older":
        attr_score = 1.0 - p_after
        attr_delta = max(0.0, p_before - p_after)
        attr_success = p_after < threshold
    else:
        attr_score = p_after
        attr_delta = max(0.0, p_after - p_before)
        attr_success = p_after >= threshold

    attr_delta_score = min(attr_delta / 0.35, 1.0)
    bg_score = (
        0.5 if background_l1 is None or not np.isfinite(background_l1)
        else 1.0 - min(float(background_l1) / 0.08, 1.0)
    )
    lp_score = (
        0.5 if lpips is None or not np.isfinite(lpips)
        else 1.0 - min(float(lpips) / 0.35, 1.0)
    )

    rerank_score = (
        weights.get("attr_score", 0.0) * attr_score
        + weights.get("attr_delta", 0.0) * attr_delta_score
        + weights.get("background_score", 0.0) * bg_score
        + weights.get("lpips_score", 0.0) * lp_score
    )

    return {
        "attr_score": float(attr_score),
        "attr_delta": float(attr_delta),
        "attr_delta_score": float(attr_delta_score),
        "attr_success": bool(attr_success),
        "background_score": float(bg_score),
        "lpips_score": float(lp_score),
        "rerank_score": float(rerank_score),
    }


def rerank(
    run_dir: Path,
    model_ids: list[str],
    classifier_path: Path,
    weights: dict[str, float],
    dry_run: bool = False,
) -> pd.DataFrame:
    model, device, thresholds = load_classifier_for_eval(classifier_path)
    rows = []

    meta_files = sorted((run_dir / "edited").glob("*/*/*_meta.json"))
    meta_files = [
        p for p in meta_files
        if p.parts[-3] in model_ids and p.parts[-2] in TASKS
    ]
    if not meta_files:
        raise FileNotFoundError("No matching *_meta.json files found.")

    for meta_path in tqdm(meta_files, desc="rerank"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task = meta["task"]
        attr_idx = TASK_TO_ATTR_INDEX[task]
        threshold = float(thresholds[attr_idx])

        orig_path = _resolve_path(meta["source_image"], meta_path)
        orig = Image.open(orig_path).convert("RGB")
        p_before = float(predict_attrs(model, device, orig)[attr_idx])

        candidate_rows = []
        for idx, candidate in enumerate(meta["candidates"]):
            path = candidate_path(candidate, meta_path)
            edited = Image.open(path).convert("RGB").resize(
                orig.size, Image.Resampling.LANCZOS
            )
            p_after = float(predict_attrs(model, device, edited)[attr_idx])
            background_l1 = candidate.get("background_l1")
            lpips = candidate.get("lpips")
            scores = score_candidate(
                task=task,
                p_before=p_before,
                p_after=p_after,
                threshold=threshold,
                background_l1=float(background_l1) if background_l1 is not None else None,
                lpips=float(lpips) if lpips is not None else None,
                weights=weights,
            )

            old_score = candidate.get("selection_score")
            candidate_rows.append({
                "candidate_index": idx,
                "candidate_path": path,
                "p_after": p_after,
                "old_selection_score": old_score,
                **scores,
            })

        best = max(candidate_rows, key=lambda r: float(r["rerank_score"]))
        old_selected = int(meta.get("selected_candidate", 0))
        new_selected = int(best["candidate_index"])

        for row in candidate_rows:
            candidate = meta["candidates"][int(row["candidate_index"])]
            if "old_selection_score" not in candidate:
                candidate["old_selection_score"] = candidate.get("selection_score")
            candidate["attr_prob_before"] = p_before
            candidate["attr_prob_after"] = row["p_after"]
            candidate["attr_score"] = row["attr_score"]
            candidate["attr_delta"] = row["attr_delta"]
            candidate["attr_delta_score"] = row["attr_delta_score"]
            candidate["attr_success"] = row["attr_success"]
            candidate["background_score"] = row["background_score"]
            candidate["lpips_score"] = row["lpips_score"]
            candidate["selection_score"] = row["rerank_score"]

        best_path = meta_path.parent / f"{meta['id']}_best.jpg"
        if not dry_run:
            Image.open(best["candidate_path"]).convert("RGB").save(best_path, quality=95)
            meta["selected_candidate"] = new_selected
            meta["selection_score"] = best["rerank_score"]
            meta["best_path"] = str(best_path)
            meta["rerank_weights"] = weights
            meta["reranked"] = True
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

        rows.append({
            "model_id": meta["model_id"],
            "task": task,
            "image_id": meta["id"],
            "old_selected": old_selected,
            "new_selected": new_selected,
            "changed": old_selected != new_selected,
            "p_before": p_before,
            "best_attr_prob_after": best["p_after"],
            "best_attr_score": best["attr_score"],
            "best_attr_delta": best["attr_delta"],
            "best_attr_success": best["attr_success"],
            "best_rerank_score": best["rerank_score"],
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank existing edited candidates.")
    parser.add_argument("--run-dir", default=None, help="Run directory.")
    parser.add_argument("--classifier-path", default=None, help="Classifier checkpoint.")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Model id to rerank. Can be repeated. Defaults to sd15 and sdxl.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated weights, e.g. attr_score=0.7,attr_delta=0.15,background_score=0.1,lpips_score=0.05",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else PATHS["run_dir"]
    classifier_path = (
        Path(args.classifier_path)
        if args.classifier_path
        else run_dir / "checkpoints" / "attr_classifier" / "best.pt"
    )
    model_ids = args.model_ids or ["sd15", "sdxl"]
    weights = parse_weights(args.weights)

    print("Rerank weights:", weights)
    report = rerank(run_dir, model_ids, classifier_path, weights, dry_run=args.dry_run)

    out_path = run_dir / "metrics" / "candidate_rerank_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")
    print(report.groupby(["model_id", "task", "best_attr_success"]).size())
    print("Changed selections:", int(report["changed"].sum()), "/", len(report))


if __name__ == "__main__":
    main()
