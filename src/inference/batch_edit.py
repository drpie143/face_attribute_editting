"""
Batch inpainting edits over the test set, with multi-candidate selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.config import (
    EDIT_CONFIG,
    MODELS,
    PATHS,
    SEED,
    TASKS,
    TARGET_PROMPTS,
    NEGATIVE_PROMPTS,
    TASK_TO_ATTR_INDEX,
    TASK_TO_HARD_MASK_KEY,
    TASK_TO_SOFT_MASK_KEY,
    load_jsonl,
)
from src.data.manifest import take_n
from src.inference.pipeline import blend_with_soft_mask, load_pipeline, unload_pipeline
from src.evaluation.metrics import background_l1, lpips_distance, identity_similarity
from src.models.classifier import load_classifier, predict_attrs


def _load_candidate_classifier(run_dir: Path):
    ckpt = run_dir / "checkpoints" / "attr_classifier" / "best.pt"
    if not ckpt.exists():
        print(
            "[WARN] Attribute classifier not found; candidate selection will "
            "use neutral attribute scores. Run `python scripts/run_training.py "
            "--classifier` before inference for attribute-aware selection."
        )
        return None, None
    try:
        model, device, _ = load_classifier(ckpt)
        print(f"[OK] loaded candidate-selection classifier: {ckpt}")
        return model, device
    except Exception as exc:
        print(f"[WARN] Could not load candidate-selection classifier: {exc}")
        return None, None


def _attribute_scores(task: str, p_before: np.ndarray, p_after: np.ndarray) -> dict[str, float]:
    idx = TASK_TO_ATTR_INDEX[task]
    before = float(p_before[idx])
    after = float(p_after[idx])
    if task == "make_older":
        attr_score = 1.0 - after
        attr_delta = max(0.0, before - after)
    else:
        attr_score = after
        attr_delta = max(0.0, after - before)
    return {
        "attr_prob_before": before,
        "attr_prob_after": after,
        "attr_score": float(attr_score),
        "attr_delta": float(attr_delta),
    }


def _score_candidate(m: Dict[str, Any]) -> float:
    """Score a single candidate using weighted criteria."""
    w = EDIT_CONFIG["selection_weights"]
    ident = 0.5 if m.get("identity_similarity") is None else float(
        np.clip((m["identity_similarity"] + 1.0) / 2.0, 0, 1))
    bg = 1.0 - min(float(m.get("background_l1", 1.0)) / 0.08, 1.0)
    lp = m.get("lpips")
    lp_score = 0.5 if lp is None else 1.0 - min(float(lp) / 0.35, 1.0)
    attr_delta = min(float(m.get("attr_delta", 0.0)) / 0.35, 1.0)
    return float(
        w["attr_score"] * m.get("attr_score", 0.5)
        + w.get("attr_delta", 0.0) * attr_delta
        + w.get("identity_score", 0.0) * ident
        + w.get("background_score", 0.0) * bg
        + w.get("lpips_score", 0.0) * lp_score
    )


def batch_edit_model(
    model_id: str,
    split_name: str = "test",
    run_dir: Optional[Path] = None,
    samples_per_task: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run batch inpainting edits for a single model over all tasks.
    Returns list of per-image metadata dicts.
    """
    run_dir = Path(run_dir or PATHS["run_dir"])
    cfg = MODELS[model_id]
    res = int(cfg["resolution"])

    pipe = load_pipeline(model_id, run_dir)
    selector, selector_device = _load_candidate_classifier(run_dir)
    recs = load_jsonl(PATHS["manifest_dir"] / f"{split_name}.jsonl")
    root = run_dir / "edited" / model_id
    root.mkdir(parents=True, exist_ok=True)

    all_meta = []
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    use_identity = EDIT_CONFIG["selection_weights"].get("identity_score", 0.0) > 0

    for task in TASKS:
        task_recs = take_n(
            [r for r in recs if task in r.get("eligible_tasks", [])],
            samples_per_task, SEED + len(task))
        task_dir = root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"[EDIT] {model_id} {task}: {len(task_recs)} images")

        for r in tqdm(task_recs, desc=f"{model_id}/{task}"):
            image_id = r["id"]
            orig = Image.open(r[f"image_{res}"]).convert("RGB")
            hard = Image.open(r[TASK_TO_HARD_MASK_KEY[task]]).convert("L").resize(
                (res, res), Image.Resampling.NEAREST)
            soft = Image.open(r[TASK_TO_SOFT_MASK_KEY[task]]).convert("L").resize(
                (res, res), Image.Resampling.BILINEAR)
            p_before = (
                predict_attrs(selector, selector_device, orig)
                if selector is not None and selector_device is not None
                else None
            )

            candidates = []
            for ci in range(EDIT_CONFIG["num_candidates_per_image"]):
                seed = SEED + ci * 1000 + int(image_id)
                gen = torch.Generator(device=gen_device).manual_seed(seed)

                with torch.inference_mode():
                    result = pipe(
                        prompt=TARGET_PROMPTS[task],
                        negative_prompt=NEGATIVE_PROMPTS[task],
                        image=orig,
                        mask_image=hard,
                        strength=EDIT_CONFIG["strength"][task][model_id],
                        guidance_scale=cfg["guidance_scale"],
                        num_inference_steps=EDIT_CONFIG.get("num_inference_steps", 30),
                        generator=gen,
                    ).images[0]

                blend = blend_with_soft_mask(orig, result, soft)
                blend_path = task_dir / f"{image_id}_candidate{ci}_blend.jpg"
                blend.save(blend_path, quality=95)

                candidate = {
                    "seed": seed,
                    "path": str(blend_path),
                    "background_l1": background_l1(orig, blend, soft),
                    "lpips": lpips_distance(orig, blend),
                    "identity_similarity": (
                        identity_similarity(orig, blend) if use_identity else None
                    ),
                }
                if p_before is not None:
                    p_after = predict_attrs(selector, selector_device, blend)
                    candidate.update(_attribute_scores(task, p_before, p_after))
                candidates.append(candidate)

            # Select best candidate
            for c in candidates:
                c["selection_score"] = _score_candidate(c)
            best_idx = int(np.argmax([c["selection_score"] for c in candidates]))
            best_path = task_dir / f"{image_id}_best.jpg"
            Image.open(candidates[best_idx]["path"]).save(best_path, quality=95)

            meta = {
                "id": image_id,
                "model_id": model_id,
                "task": task,
                "engine": EDIT_CONFIG["engine"],
                "source_image": r[f"image_{res}"],
                "best_path": str(best_path),
                "mask_soft": r[TASK_TO_SOFT_MASK_KEY[task]],
                "mask_hard": r[TASK_TO_HARD_MASK_KEY[task]],
                "strength": EDIT_CONFIG["strength"][task][model_id],
                "guidance_scale": cfg["guidance_scale"],
                "selected_candidate": best_idx,
                "candidates": candidates,
            }
            meta_path = task_dir / f"{image_id}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            all_meta.append(meta)

    unload_pipeline(pipe)
    return all_meta
