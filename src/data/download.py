"""
Download raw CelebA / CelebAMask-HQ datasets via the Kaggle API.

Functions handle credential setup, downloading, unzipping, and
normalising the raw directory layout so that downstream preprocessing
always sees a canonical tree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import CFG, PATHS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def first_existing(paths: List[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def dir_has_files(path: Path, max_check: int = 1) -> bool:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False
    count = 0
    for _ in path.rglob("*"):
        count += 1
        if count >= max_check:
            return True
    return False


def get_mask_anno_dir() -> Optional[Path]:
    raw_maskhq = PATHS["raw_maskhq"]
    return first_existing([
        raw_maskhq / "CelebAMask-HQ-mask-anno",
        raw_maskhq / "CelebA-HQ-mask-anno",
    ])


def raw_data_status() -> Dict[str, bool]:
    raw_maskhq = PATHS["raw_maskhq"]
    raw_celeba = PATHS["raw_celeba"]
    mask_dir = get_mask_anno_dir()
    return {
        "hq_img_dir": dir_has_files(raw_maskhq / "CelebA-HQ-img"),
        "mask_anno_dir": mask_dir is not None and dir_has_files(mask_dir),
        "mapping_path": (raw_maskhq / "CelebA-HQ-to-CelebA-mapping.txt").exists(),
        "attr_path": first_existing([
            raw_celeba / "list_attr_celeba.csv",
            raw_celeba / "list_attr_celeba.txt",
        ]) is not None,
        "partition_path": first_existing([
            raw_celeba / "list_eval_partition.csv",
            raw_celeba / "list_eval_partition.txt",
        ]) is not None,
    }


def has_all_raw_data() -> bool:
    return all(raw_data_status().values())


def print_raw_data_status(prefix: str = "") -> None:
    status = raw_data_status()
    print(f"{prefix}Raw data status:")
    for k, ok in status.items():
        print(f"  {'[OK]' if ok else '[MISSING]'} {k}")


# ---------------------------------------------------------------------------
# CelebA txt to csv normalisation
# ---------------------------------------------------------------------------
def normalize_celeba_txt_to_csv() -> None:
    """Convert official CelebA .txt annotation files to .csv."""
    raw_celeba = PATHS["raw_celeba"]
    attr_txt = raw_celeba / "list_attr_celeba.txt"
    attr_csv = raw_celeba / "list_attr_celeba.csv"
    part_txt = raw_celeba / "list_eval_partition.txt"
    part_csv = raw_celeba / "list_eval_partition.csv"

    if attr_txt.exists() and not attr_csv.exists():
        with open(attr_txt, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
            cols = f.readline().strip().split()
        if first.isdigit() and cols:
            df = pd.read_csv(attr_txt, sep=r"\s+", skiprows=2,
                             names=["image_id"] + cols)
            df.to_csv(attr_csv, index=False)
            print(f"[OK] converted {attr_txt.name} -> {attr_csv.name}")

    if part_txt.exists() and not part_csv.exists():
        df = pd.read_csv(part_txt, sep=r"\s+", header=None,
                         names=["image_id", "partition"])
        df.to_csv(part_csv, index=False)
        print(f"[OK] converted {part_txt.name} -> {part_csv.name}")


# ---------------------------------------------------------------------------
# Kaggle credentials
# ---------------------------------------------------------------------------
def ensure_kaggle_credentials() -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    if kaggle_json.exists():
        os.chmod(kaggle_json, 0o600)
        return True

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        kaggle_json.write_text(
            json.dumps({"username": env_user, "key": env_key}),
            encoding="utf-8",
        )
        os.chmod(kaggle_json, 0o600)
        print("[OK] created Kaggle credential from env vars")
        return True

    print("[WARN] Kaggle credentials not found. Set KAGGLE_USERNAME / KAGGLE_KEY.")
    return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def kaggle_download_dataset(slug: str, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"]
    print("[CMD]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print("[WARN] Download failed:", repr(e))
        return False


def find_path_by_name(root: Path, names: List[str],
                      want_dir: Optional[bool] = None) -> Optional[Path]:
    if not root.exists():
        return None
    wanted = set(names)
    for p in root.rglob("*"):
        if p.name in wanted:
            if want_dir is True and not p.is_dir():
                continue
            if want_dir is False and not p.is_file():
                continue
            return p
    return None


def download_if_needed(download_dir: Optional[Path] = None) -> None:
    """Download CelebA + CelebAMask-HQ via Kaggle API if raw data is missing."""
    if has_all_raw_data():
        print("[OK] Raw data already present; skipping download.")
        return

    if not ensure_kaggle_credentials():
        print_raw_data_status()
        return

    dl_dir = download_dir or (PATHS["project_root"] / "dataset_downloads")
    raw_maskhq = PATHS["raw_maskhq"]
    raw_celeba = PATHS["raw_celeba"]
    raw_maskhq.mkdir(parents=True, exist_ok=True)
    raw_celeba.mkdir(parents=True, exist_ok=True)

    data_cfg = CFG["data"]

    # CelebAMask-HQ
    cmhq_dir = dl_dir / "celebamaskhq"
    kaggle_download_dataset(data_cfg["kaggle_celebamaskhq"], cmhq_dir)
    for name, dst in [
        ("CelebA-HQ-img", raw_maskhq / "CelebA-HQ-img"),
        ("CelebAMask-HQ-mask-anno", None),  # handled below
        ("CelebA-HQ-to-CelebA-mapping.txt", raw_maskhq / "CelebA-HQ-to-CelebA-mapping.txt"),
    ]:
        src = find_path_by_name(cmhq_dir, [name],
                                want_dir=(name != "CelebA-HQ-to-CelebA-mapping.txt"))
        if src and dst and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            print(f"[OK] {name} -> {dst}")

    # Handle mask dir name variants
    for mask_name in ["CelebAMask-HQ-mask-anno", "CelebA-HQ-mask-anno"]:
        src = find_path_by_name(cmhq_dir, [mask_name], want_dir=True)
        if src:
            dst = raw_maskhq / src.name
            if not dst.exists():
                shutil.copytree(src, dst)
                print(f"[OK] {mask_name} -> {dst}")
            break

    # CelebA attributes
    celeba_dir = dl_dir / "celeba"
    kaggle_download_dataset(data_cfg["kaggle_celeba"], celeba_dir)
    for name in ["list_attr_celeba.csv", "list_eval_partition.csv"]:
        src = find_path_by_name(celeba_dir, [name], want_dir=False)
        if src:
            dst = raw_celeba / name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"[OK] {name} -> {dst}")

    normalize_celeba_txt_to_csv()
    print_raw_data_status("[AFTER DOWNLOAD] ")
