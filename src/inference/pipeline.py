"""
Inpainting pipeline management: load base model + LoRA, run inference.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.config import MODELS, PATHS


def load_pipeline(model_id: str, run_dir: Optional[Path] = None):
    """
    Load an AutoPipelineForInpainting with optional LoRA weights.
    Returns the pipeline on CUDA (or CPU fallback).
    """
    from diffusers import AutoPipelineForInpainting

    cfg = MODELS[model_id]
    base = cfg["pretrained_model_name_or_path"]
    run_dir = Path(run_dir or PATHS["run_dir"])
    lora_dir = run_dir / "loras" / model_id

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    kwargs = dict(torch_dtype=dtype, use_safetensors=True)
    try:
        if has_cuda:
            pipe = AutoPipelineForInpainting.from_pretrained(
                base, variant="fp16", **kwargs)
        else:
            pipe = AutoPipelineForInpainting.from_pretrained(base, **kwargs)
    except Exception:
        pipe = AutoPipelineForInpainting.from_pretrained(base, **kwargs)

    # Check if run directory contains LoRA weights
    loaded_lora = False
    if lora_dir.exists() and any(lora_dir.glob("*.safetensors")):
        try:
            pipe.load_lora_weights(str(lora_dir))
            print(f"[OK] loaded LoRA from run_dir: {lora_dir}")
            loaded_lora = True
        except Exception as e:
            print(f"[WARN] could not load LoRA from run_dir: {e}")

    # Fallback to exports/best_loras/
    if not loaded_lora:
        fallback_path = PATHS["exports_dir"] / "best_loras" / f"{model_id}_faceattr_lora.safetensors"
        if fallback_path.exists():
            try:
                pipe.load_lora_weights(str(fallback_path))
                print(f"[OK] loaded fallback best LoRA: {fallback_path}")
                loaded_lora = True
            except Exception as e:
                print(f"[WARN] could not load fallback LoRA: {e}")

    if not loaded_lora:
        print("[WARN] No LoRA weights found, using base model only.")

    device = "cuda" if has_cuda else "cpu"
    pipe.to(device)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


def blend_with_soft_mask(original: Image.Image, edited: Image.Image,
                         mask: Image.Image) -> Image.Image:
    """Composite edited region onto original using a soft mask."""
    return Image.composite(
        edited.convert("RGB").resize(original.size, Image.Resampling.LANCZOS),
        original.convert("RGB"),
        mask.convert("L").resize(original.size, Image.Resampling.BILINEAR),
    )


def unload_pipeline(pipe) -> None:
    """Release a pipeline and free GPU memory."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
