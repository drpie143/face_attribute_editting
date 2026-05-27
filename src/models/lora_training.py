"""
LoRA training launcher using the official Diffusers text-to-image scripts.
"""

from __future__ import annotations

import gc
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.config import (
    MODELS,
    PATHS,
    SEED,
    load_jsonl,
    train_steps_for_model,
    warmup_steps_for_model,
)


def build_diffusers_imagefolder(
    model_id: str,
    split_name: str = "train",
    force: bool = False,
) -> Path:
    """Build the image folder with metadata.jsonl required by Diffusers."""
    cfg = MODELS[model_id]
    res = int(cfg["resolution"])
    
    local_dir = PATHS["processed_dir"] / "diffusers" / model_id
    cache_dir = PATHS["processed_dir"] / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / f"diffusers_{model_id}_{split_name}_{res}.tar"

    if local_dir.exists() and (local_dir / "metadata.jsonl").exists() and not force:
        print(f"[SKIP] existing local imagefolder for {model_id}: {local_dir}")
        return local_dir

    if tar_path.exists() and not force:
        if local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[EXTRACT] {tar_path} -> {local_dir.parent}")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(local_dir.parent)
        return local_dir

    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = PATHS["manifest_dir"] / f"{split_name}.jsonl"
    recs = load_jsonl(manifest_path)
    metadata = []
    for r in tqdm(recs, desc=f"copy {model_id} imagefolder"):
        src = Path(r[f"image_{res}"])
        dst_name = f"{r['id']}.jpg"
        dst = local_dir / dst_name
        shutil.copy2(src, dst)
        metadata.append({"file_name": dst_name, "text": r["caption"]})

    with open(local_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[DONE] {model_id} imagefolder created with {len(metadata)} items at {local_dir}")

    print(f"[TAR CACHE] writing to {tar_path}")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(local_dir, arcname=local_dir.name)

    return local_dir



# ===== Checkpoint helpers ===================================================

def has_checkpoint(output_dir: Path) -> bool:
    return output_dir.exists() and any(
        p.name.startswith("checkpoint-") for p in output_dir.iterdir() if p.is_dir())


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    ckpts = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=_checkpoint_step)
    return ckpts[-1] if ckpts else None


def final_lora_weight(output_dir: Path) -> Optional[Path]:
    for name in ["pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"]:
        p = output_dir / name
        if p.exists():
            return p
    if output_dir.exists():
        hits = sorted(list(output_dir.glob("*.safetensors")) +
                       list(output_dir.glob("*.bin")))
        return hits[0] if hits else None
    return None


def cleanup_runtime_memory(note: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[CLEANUP] released cache{': ' + note if note else ''}")


# ===== Command builder ======================================================

def build_train_command(
    model_id: str,
    data_dir: Path,
    run_dir: Path,
    vendor_diffusers_dir: Path,
    smoke: bool = False,
    checkpointing_steps: int = 500,
    num_workers: int = 4,
) -> List[str]:
    """Build the accelerate launch command for LoRA training."""
    cfg = MODELS[model_id]
    is_sd15 = cfg["architecture"] == "sd15"
    script_name = ("train_text_to_image_lora.py" if is_sd15
                    else "train_text_to_image_lora_sdxl.py")
    script = vendor_diffusers_dir / "examples" / "text_to_image" / script_name

    output_dir = run_dir / "loras" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = run_dir / "logs" / f"{model_id}_diffusers"
    logging_dir.mkdir(parents=True, exist_ok=True)

    steps = train_steps_for_model(model_id, smoke)
    warmup = warmup_steps_for_model(model_id, smoke)

    cmd = [
        "accelerate", "launch", str(script),
        f"--pretrained_model_name_or_path={cfg['pretrained_model_name_or_path']}",
        "--dataset_name=imagefolder",
        f"--train_data_dir={data_dir}",
        "--caption_column=text",
        f"--resolution={cfg['resolution']}",
        "--center_crop", "--random_flip",
        f"--train_batch_size={cfg['train_batch_size']}",
        f"--gradient_accumulation_steps={cfg['gradient_accumulation_steps']}",
        "--gradient_checkpointing",
        f"--max_train_steps={steps}",
        f"--learning_rate={cfg['learning_rate']}",
        "--lr_scheduler=cosine",
        f"--lr_warmup_steps={warmup}",
        "--snr_gamma=5.0",
        f"--rank={cfg['rank']}",
        "--mixed_precision=fp16",
        f"--checkpointing_steps={checkpointing_steps}",
        "--checkpoints_total_limit=4",
        "--validation_prompt=a high quality realistic face portrait, "
        "wearing eyeglasses, smiling, natural skin texture, sharp facial details",
        "--validation_epochs=1",
        f"--seed={SEED}",
        f"--output_dir={output_dir}",
        f"--logging_dir={logging_dir}",
        "--report_to=tensorboard",
        f"--dataloader_num_workers={num_workers}",
    ]

    if has_checkpoint(output_dir):
        cmd.append("--resume_from_checkpoint=latest")

    return cmd


# ===== Training runner ======================================================

def run_lora_training(
    model_id: str,
    data_dir: Path,
    run_dir: Optional[Path] = None,
    vendor_diffusers_dir: Optional[Path] = None,
    smoke: bool = False,
    force: bool = False,
) -> Path:
    """
    Train a LoRA adapter for the given model. Returns the output directory.
    """
    run_dir = Path(run_dir or PATHS["run_dir"])
    output_dir = run_dir / "loras" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = train_steps_for_model(model_id, smoke)

    # Skip if already trained
    existing = final_lora_weight(output_dir)
    if existing is not None and not force:
        print(f"[SKIP] existing LoRA for {model_id}: {existing}")
        return output_dir

    # Default vendor dir
    if vendor_diffusers_dir is None:
        vendor_diffusers_dir = PATHS["project_root"] / "vendor" / "diffusers"

    subprocess.run(["accelerate", "config", "default"], check=False)

    cmd = build_train_command(model_id, data_dir, run_dir, vendor_diffusers_dir,
                              smoke=smoke)
    log_path = run_dir / "logs" / f"{model_id}_lora_train_stdout.log"
    print(f"[TRAIN] {model_id}: steps={steps}, log={log_path}")

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log_f.write(line)
            if any(k in line for k in ["loss=", "Saving", "checkpoint", "100%"]):
                print(line, end="")
        ret = proc.wait()

    elapsed = round((time.time() - started) / 60.0, 2)
    status = "completed" if ret == 0 else "failed"
    print(f"[{status.upper()}] {model_id}: elapsed={elapsed} min")

    if ret != 0:
        raise RuntimeError(f"LoRA training failed for {model_id}. Check: {log_path}")

    # Export weight
    exports_dir = PATHS["exports_dir"] / "best_loras"
    exports_dir.mkdir(parents=True, exist_ok=True)
    src = final_lora_weight(output_dir)
    if src:
        dst = exports_dir / f"{model_id}_faceattr_lora{src.suffix}"
        shutil.copy2(src, dst)
        print(f"[EXPORT] {src} -> {dst}")

    cleanup_runtime_memory(model_id)
    return output_dir
