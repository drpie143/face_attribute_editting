"""
Manifest (train/val/test split) building and I/O.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from src.config import ATTR_COLUMNS, PATHS, SEED, load_jsonl, save_jsonl


def split_records(records: List[Dict[str, Any]],
                  seed: int = SEED):
    """Split records into train/val/test by CelebA partition or random 80/10/10."""
    has_part = any(r.get("partition", -1) in [0, 1, 2] for r in records)
    if has_part:
        train = [r for r in records if r.get("partition") == 0]
        val = [r for r in records if r.get("partition") == 1]
        test = [r for r in records if r.get("partition") == 2]
        if not val or not test:
            has_part = False
    if not has_part:
        rng = random.Random(seed)
        shuffled = list(records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
    return train, val, test


def _record_stratum(r: Dict[str, Any]) -> str:
    attrs = r.get("attributes", {})
    attr_key = "".join("1" if attrs.get(c, False) else "0" for c in ATTR_COLUMNS)
    task_key = "+".join(sorted(r.get("eligible_tasks", [])))
    return f"{attr_key}|{task_key}"


def take_n(records: List[Dict[str, Any]], n: int, seed: int = SEED):
    """Stratified sampling of at most n records."""
    rng = random.Random(seed)
    groups: Dict[str, List] = {}
    for r in records:
        groups.setdefault(_record_stratum(r), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = list(groups.keys())
    rng.shuffle(keys)
    selected = []
    while len(selected) < min(n, len(records)) and keys:
        next_keys = []
        for key in keys:
            if len(selected) >= min(n, len(records)):
                break
            if groups[key]:
                selected.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def build_manifests(records: List[Dict[str, Any]],
                    seed: int = SEED,
                    force: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Build and save train/val/test + smoke manifests."""
    manifest_dir = PATHS["manifest_dir"]
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if (manifest_dir / "train.jsonl").exists() and not force:
        splits = {}
        for name in ["train", "val", "test", "train_smoke", "val_smoke", "test_smoke"]:
            splits[name] = load_jsonl(manifest_dir / f"{name}.jsonl")
        print("[SKIP] Loaded existing manifests")
        return splits

    train, val, test = split_records(records, seed)
    splits = {
        "train": train,
        "val": val,
        "test": test,
        "train_smoke": take_n(train, 128, seed),
        "val_smoke": take_n(val, 32, seed + 1),
        "test_smoke": take_n(test, 32, seed + 2),
    }
    for name, recs in splits.items():
        save_jsonl(recs, manifest_dir / f"{name}.jsonl")
        print(f"[SAVED] {name}: {len(recs)} records")

    return splits
