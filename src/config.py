"""
Global configuration, path management, and utility helpers.

This module centralises all project paths, model definitions, and shared
constants so that every other module can simply ``from src.config import ...``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
IS_KAGGLE = Path("/kaggle/working").exists()

def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False

IS_COLAB = running_in_colab()

# ---------------------------------------------------------------------------
# Load YAML config
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the project YAML config."""
    p = Path(path) if path else _CFG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()

# ---------------------------------------------------------------------------
# Core constants (from YAML)
# ---------------------------------------------------------------------------
SEED: int = CFG["seed"]
RUN_TAG: str = CFG["run_tag"]
TASKS: List[str] = CFG["tasks"]
MODELS: Dict[str, Any] = CFG["models"]
EDIT_CONFIG: Dict[str, Any] = CFG["edit"]
CLASSIFIER_CONFIG: Dict[str, Any] = CFG["classifier"]
MASK_CONFIG: Dict[str, Any] = CFG["mask"]
ATTR_COLUMNS: List[str] = CFG["data"]["attr_columns"]
RESOLUTIONS: List[int] = CFG["data"]["resolutions"]

# Prompts
TARGET_PROMPTS: Dict[str, str] = CFG["prompts"]["target"]
NEGATIVE_PROMPTS: Dict[str, str] = CFG["prompts"]["negative"]

# Task to attribute index mapping
TASK_TO_ATTR_INDEX = {"add_eyeglasses": 0, "make_smiling": 1, "make_older": 2}

# Task to mask key mapping
TASK_TO_HARD_MASK_KEY = {
    "add_eyeglasses": "mask_eyeglasses_hard",
    "make_smiling": "mask_smile_hard",
    "make_older": "mask_age_hard",
}
TASK_TO_SOFT_MASK_KEY = {
    "add_eyeglasses": "mask_eyeglasses_soft",
    "make_smiling": "mask_smile_soft",
    "make_older": "mask_age_soft",
}

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
def setup_paths(project_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Return a dict of canonical project paths. Adapts to Kaggle / Colab / local.
    """
    if project_root is not None:
        base = Path(project_root)
    elif IS_KAGGLE:
        base = Path("/kaggle/working/face-attr-edit")
    elif IS_COLAB:
        base = Path("/content/drive/MyDrive/face-attr-edit")
    else:
        base = Path(__file__).resolve().parent.parent

    raw_dir = base / "data" / "raw"
    processed_dir = base / "data" / "processed"
    runs_dir = base / "runs"
    exports_dir = base / "exports"

    return {
        "project_root": base,
        "raw_dir": raw_dir,
        "raw_celeba": raw_dir / "celeba",
        "raw_maskhq": raw_dir / "celebamask_hq",
        "processed_dir": processed_dir,
        "manifest_dir": processed_dir / "manifests",
        "runs_dir": runs_dir,
        "run_dir": runs_dir / RUN_TAG,
        "exports_dir": exports_dir,
        "configs_dir": base / "configs",
    }

PATHS = setup_paths()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_steps_for_model(model_id: str, smoke: bool = False) -> int:
    cfg = MODELS[model_id]
    return 30 if smoke else int(cfg["max_train_steps"])


def warmup_steps_for_model(model_id: str, smoke: bool = False) -> int:
    steps = train_steps_for_model(model_id, smoke)
    return 0 if steps < 100 else max(10, int(steps * 0.03))
