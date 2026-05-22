"""
Image preprocessing and semantic-mask generation for CelebAMask-HQ.

Steps:
  1. Read HQ-to-CelebA mapping, attributes, and partitions.
  2. Resize images to 512 / 768 / 1024.
  3. Build task-specific masks (eyeglasses, smile, age) from semantic parts.
  4. Produce per-image records for downstream manifest building.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from tqdm.auto import tqdm

from src.config import (
    ATTR_COLUMNS,
    MASK_CONFIG,
    PATHS,
    RESOLUTIONS,
    load_jsonl,
    save_jsonl,
)
from src.data.download import first_existing, get_mask_anno_dir


# ===== I/O helpers ==========================================================

def read_celeba_attr(path: Path) -> pd.DataFrame:
    """Read CelebA attribute annotations (csv or txt)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        second = f.readline().strip()

    if first.isdigit() and "Eyeglasses" in second:
        names = second.split()
        df = pd.read_csv(path, sep=r"\s+", skiprows=2,
                         names=["image_id"] + names)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=r"\s+")

    if "image_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "image_id"})
    df["image_id"] = df["image_id"].astype(str)
    keep = ["image_id"] + [c for c in ATTR_COLUMNS if c in df.columns]
    return df[keep].copy()


def read_celeba_partition(path: Path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", header=None)
    if len(df.columns) == 2:
        df.columns = ["image_id", "partition"]
    elif "image_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "image_id",
                                df.columns[1]: "partition"})
    df["image_id"] = df["image_id"].astype(str)
    df["partition"] = df["partition"].astype(int)
    return df[["image_id", "partition"]].copy()


def read_hq_mapping(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if "idx" in low and ("orig" in low or "image" in low):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                hq_index = int(parts[0])
            except Exception:
                continue
            celeba_file = None
            for part in reversed(parts[1:]):
                if part.lower().endswith((".jpg", ".png", ".jpeg")):
                    celeba_file = part
                    break
            if celeba_file is None and len(parts) >= 3:
                try:
                    orig_idx = int(parts[1])
                    celeba_file = f"{orig_idx + 1:06d}.jpg"
                except Exception:
                    celeba_file = str(parts[-1])
            elif celeba_file is None:
                celeba_file = str(parts[-1])
            rows.append({"hq_index": hq_index, "celeba_file": celeba_file})
    df = pd.DataFrame(rows).drop_duplicates("hq_index")
    df["celeba_file"] = df["celeba_file"].astype(str)
    return df.sort_values("hq_index").reset_index(drop=True)


def load_metadata() -> pd.DataFrame:
    """Merge mapping + attributes + partitions into a single DataFrame."""
    raw_maskhq = PATHS["raw_maskhq"]
    raw_celeba = PATHS["raw_celeba"]

    attr_path = first_existing([
        raw_celeba / "list_attr_celeba.csv",
        raw_celeba / "list_attr_celeba.txt",
    ])
    part_path = first_existing([
        raw_celeba / "list_eval_partition.csv",
        raw_celeba / "list_eval_partition.txt",
    ])
    mapping_path = raw_maskhq / "CelebA-HQ-to-CelebA-mapping.txt"

    attrs_df = read_celeba_attr(attr_path)
    part_df = read_celeba_partition(part_path)
    mapping_df = read_hq_mapping(mapping_path)

    meta = mapping_df.merge(attrs_df, left_on="celeba_file",
                            right_on="image_id", how="left")
    meta = meta.merge(part_df, left_on="celeba_file",
                      right_on="image_id", how="left",
                      suffixes=("", "_part"))
    meta = meta.drop(columns=[c for c in ["image_id", "image_id_part"]
                               if c in meta.columns])
    return meta


# ===== Mask building ========================================================

def _find_hq_image(hq_index: int) -> Optional[Path]:
    img_dir = PATHS["raw_maskhq"] / "CelebA-HQ-img"
    for pat in [f"{hq_index}.jpg", f"{hq_index:05d}.jpg", f"{hq_index}.png"]:
        p = img_dir / pat
        if p.exists():
            return p
    return None


def _load_binary_mask(hq_index: int, part_names: List[str],
                      size: int = 512) -> Image.Image:
    mask_dir = get_mask_anno_dir()
    acc = np.zeros((size, size), dtype=np.uint8)
    for name in part_names:
        roots = [mask_dir]
        sub = mask_dir / str(hq_index // 2000)
        if sub.exists():
            roots.insert(0, sub)
        for root in roots:
            for pat in [f"{hq_index:05d}_{name}.png", f"{hq_index}_{name}.png"]:
                p = root / pat
                if p.exists():
                    try:
                        a = np.array(Image.open(p).convert("L").resize(
                            (size, size), Image.Resampling.NEAREST))
                        acc = np.maximum(acc, (a > 0).astype(np.uint8) * 255)
                    except Exception:
                        pass
                    break
    return Image.fromarray(acc)


def _arr(mask: Image.Image) -> np.ndarray:
    return np.array(mask.convert("L"), dtype=np.uint8)


def _to_mask(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _morph(mask: Image.Image, kernel: int = 5,
           op: str = "dilate", iterations: int = 1) -> Image.Image:
    a = _arr(mask)
    if a.max() == 0:
        return mask.convert("L")
    k = np.ones((kernel, kernel), np.uint8)
    ops = {
        "erode": lambda: cv2.erode(a, k, iterations=iterations),
        "close": lambda: cv2.morphologyEx(a, cv2.MORPH_CLOSE, k, iterations=iterations),
        "open": lambda: cv2.morphologyEx(a, cv2.MORPH_OPEN, k, iterations=iterations),
        "dilate": lambda: cv2.dilate(a, k, iterations=iterations),
    }
    out = ops.get(op, ops["dilate"])()
    return _to_mask((out > 0).astype(np.uint8) * 255)


def _soften_mask(mask: Image.Image, dilate_kernel: int = 11,
                 blur_radius: int = 7) -> Image.Image:
    a = _arr(mask)
    if a.max() == 0:
        return mask.convert("L")
    a = cv2.dilate(a, np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=1)
    return Image.fromarray(a).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)).convert("L")


def _mask_bbox(mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(_arr(mask) > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _clamp_box(box, size=512):
    x0, y0, x1, y1 = box
    x0 = int(max(0, min(size - 1, round(x0))))
    y0 = int(max(0, min(size - 1, round(y0))))
    x1 = int(max(x0 + 1, min(size, round(x1))))
    y1 = int(max(y0 + 1, min(size, round(y1))))
    return x0, y0, x1, y1


def build_eyeglasses_mask(hq_index: int, size: int = 512) -> Image.Image:
    eyes = _load_binary_mask(hq_index, ["l_eye", "r_eye"], size)
    brows = _load_binary_mask(hq_index, ["l_brow", "r_brow"], size)
    combined = _to_mask(np.maximum(_arr(eyes), _arr(brows)))
    box = _mask_bbox(combined)
    if box is None:
        return Image.new("L", (size, size), 0)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    total_w = max((x1 - x0) * 1.55, size * 0.34)
    lens_w = total_w * 0.42
    lens_h = max((y1 - y0) * 2.25, size * 0.105)
    gap = total_w * 0.06
    left = _clamp_box((cx - gap / 2 - lens_w, cy - lens_h * 0.48,
                        cx - gap / 2, cy + lens_h * 0.62), size)
    right = _clamp_box((cx + gap / 2, cy - lens_h * 0.48,
                         cx + gap / 2 + lens_w, cy + lens_h * 0.62), size)
    bridge = _clamp_box((left[2] - gap * 0.45, cy - lens_h * 0.13,
                          right[0] + gap * 0.45, cy + lens_h * 0.17), size)
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    r = max(2, int(min(left[2] - left[0], left[3] - left[1]) * 0.35))
    d.rounded_rectangle(list(left), radius=r, fill=255)
    d.rounded_rectangle(list(right), radius=r, fill=255)
    d.rounded_rectangle(list(bridge), radius=max(1, int(lens_h * 0.10)), fill=255)
    return _morph(_to_mask(np.maximum(_arr(m), _arr(_morph(eyes, 11)))), 5, "close")


def build_smile_mask(hq_index: int, size: int = 512) -> Image.Image:
    mouth = _load_binary_mask(hq_index,
                              ["mouth", "u_lip", "l_lip", "upper_lip", "lower_lip"], size)
    box = _mask_bbox(mouth)
    if box is None:
        return Image.new("L", (size, size), 0)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * 2.10, size * 0.24)
    h = max((y1 - y0) * 2.65, size * 0.105)
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse(
        list(_clamp_box((cx - w / 2, cy - h * 0.45, cx + w / 2, cy + h * 0.62), size)),
        fill=255)
    return _morph(_to_mask(np.maximum(_arr(m), _arr(_morph(mouth, 17)))), 7, "close")


def build_age_mask(hq_index: int, size: int = 512) -> Image.Image:
    skin = _load_binary_mask(hq_index, ["skin", "nose"], size)
    if _mask_bbox(skin) is None:
        return Image.new("L", (size, size), 0)
    protected = _load_binary_mask(hq_index, [
        "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip",
        "upper_lip", "lower_lip", "hair", "hat", "neck",
    ], size)
    a = _arr(_morph(skin, 9, "close"))
    a[_arr(_morph(protected, 13)) > 0] = 0
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    return _to_mask((a > 0).astype(np.uint8) * 255)


# ===== Caption builder ======================================================

def build_caption(attrs: Dict[str, bool]) -> str:
    age = "young adult" if attrs["Young"] else "older adult"
    glasses = "wearing eyeglasses" if attrs["Eyeglasses"] else "without eyeglasses"
    smile = "smiling" if attrs["Smiling"] else "neutral expression, not smiling"
    return (f"a high quality realistic face portrait, {age}, {glasses}, "
            f"{smile}, natural skin texture, sharp facial details")


# ===== Main preprocessing ===================================================

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    return img.crop(((w - side) // 2, (h - side) // 2,
                     (w + side) // 2, (h + side) // 2))


def preprocess_one(row: pd.Series, processed_dir: Path,
                   force: bool = False) -> Optional[Dict[str, Any]]:
    """Process a single HQ image: resize, build masks, produce a record dict."""
    hq_index = int(row["hq_index"])
    if any(pd.isna(row[c]) for c in ATTR_COLUMNS):
        return None

    img_path = _find_hq_image(hq_index)
    if img_path is None:
        return None

    try:
        img = center_crop_square(Image.open(img_path).convert("RGB"))
    except Exception:
        return None

    out_id = f"{hq_index:05d}"
    image_paths = {}
    for res in RESOLUTIONS:
        out_dir = processed_dir / f"images_{res}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{out_id}.jpg"
        if force or not out_path.exists():
            img.resize((res, res), Image.Resampling.LANCZOS).save(out_path, quality=95)
        image_paths[f"image_{res}"] = str(out_path)

    attrs_bool = {
        "Eyeglasses": int(row["Eyeglasses"]) == 1,
        "Smiling": int(row["Smiling"]) == 1,
        "Young": int(row["Young"]) == 1,
    }

    # Masks
    hard_dir = processed_dir / MASK_CONFIG["hard_dir"]
    soft_dir = processed_dir / MASK_CONFIG["soft_dir"]
    hard_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)

    mask_builders = {
        "eyeglasses": build_eyeglasses_mask,
        "smile": build_smile_mask,
        "age": build_age_mask,
    }
    soft_params = {"eyeglasses": (9, 5), "smile": (15, 9), "age": (11, 11)}

    mask_paths: Dict[str, str] = {}
    coverage_range = MASK_CONFIG["coverage_range"]
    eligible = []

    for name, builder in mask_builders.items():
        hard_mask = builder(hq_index)
        soft_mask = _soften_mask(hard_mask, *soft_params[name])
        hp = hard_dir / f"{out_id}_{name}_hard.png"
        sp = soft_dir / f"{out_id}_{name}_soft.png"
        if force or not hp.exists():
            hard_mask.save(hp)
        if force or not sp.exists():
            soft_mask.save(sp)
        mask_paths[f"mask_{name}_hard"] = str(hp)
        mask_paths[f"mask_{name}_soft"] = str(sp)

        # Check coverage validity
        coverage = float((_arr(hard_mask) > 0).mean())
        lo, hi = coverage_range[name]
        valid = _mask_bbox(hard_mask) is not None and lo <= coverage <= hi

        if name == "eyeglasses" and not attrs_bool["Eyeglasses"] and valid:
            eligible.append("add_eyeglasses")
        elif name == "smile" and not attrs_bool["Smiling"] and valid:
            eligible.append("make_smiling")
        elif name == "age" and attrs_bool["Young"] and valid:
            eligible.append("make_older")

    if not eligible:
        return None

    caption = build_caption(attrs_bool)
    partition = int(row["partition"]) if not pd.isna(row.get("partition", np.nan)) else -1

    return {
        "id": out_id,
        "hq_index": hq_index,
        "partition": partition,
        **image_paths,
        **mask_paths,
        "mask_eyeglasses": mask_paths["mask_eyeglasses_soft"],
        "mask_smile": mask_paths["mask_smile_soft"],
        "mask_age": mask_paths["mask_age_soft"],
        "attributes": attrs_bool,
        "caption": caption,
        "eligible_tasks": eligible,
    }


def run_preprocessing(max_images: int = 2000, force: bool = False) -> List[Dict[str, Any]]:
    """
    Run full preprocessing pipeline: read metadata, resize images, build masks.
    Returns a list of record dicts.
    """
    processed_dir = PATHS["processed_dir"]
    records_path = processed_dir / "processed_records.jsonl"

    if records_path.exists() and not force:
        records = load_jsonl(records_path)
        if records:
            print(f"[SKIP] Loaded {len(records)} existing records from {records_path}")
            return records

    meta_df = load_metadata()
    subset = meta_df.head(max_images)
    print(f"[PREPROCESS] Processing {len(subset)} images...")

    records = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="preprocess"):
        rec = preprocess_one(row, processed_dir, force=force)
        if rec is not None:
            records.append(rec)

    save_jsonl(records, records_path)
    print(f"[DONE] {len(records)} valid records saved to {records_path}")
    return records
