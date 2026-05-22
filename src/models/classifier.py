"""
EfficientNet-B0 multi-label attribute classifier for face attributes.

Predicts: Eyeglasses, Smiling, Young — used both for candidate selection
during editing and for evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm.auto import tqdm

from src.config import ATTR_COLUMNS, CLASSIFIER_CONFIG, PATHS, SEED, load_jsonl


# ===== Dataset ==============================================================

class FaceAttrDataset(Dataset):
    """PyTorch dataset for face attribute classification."""

    def __init__(self, records: List[Dict[str, Any]],
                 image_key: str = "image_512",
                 image_size: int = 224, train: bool = True):
        self.records = records
        self.image_key = image_key
        aug = [transforms.Resize((image_size, image_size))]
        if train:
            aug += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08,
                                       saturation=0.06, hue=0.02),
            ]
        aug += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        self.tf = transforms.Compose(aug)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = self.tf(Image.open(r[self.image_key]).convert("RGB"))
        y = torch.tensor([
            float(r["attributes"]["Eyeglasses"]),
            float(r["attributes"]["Smiling"]),
            float(r["attributes"]["Young"]),
        ], dtype=torch.float32)
        return x, y, r["id"]


# ===== Model ================================================================

def build_attr_classifier(use_pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B0 with a 3-output head."""
    weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, len(ATTR_COLUMNS))
    return model


# ===== Training helpers =====================================================

def _compute_pos_weight(records, device):
    y = np.array([
        [float(r["attributes"]["Eyeglasses"]),
         float(r["attributes"]["Smiling"]),
         float(r["attributes"]["Young"])]
        for r in records
    ], dtype=np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return torch.tensor(
        np.clip(neg / np.maximum(pos, 1.0), 0.25, 8.0),
        dtype=torch.float32, device=device)


def _collect_outputs(model, loader, device):
    model.eval()
    ys, ps = [], []
    total_loss, total = 0.0, 0
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += bce(logits, y).item()
            total += y.shape[0]
            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    if not ys:
        return np.zeros((0, len(ATTR_COLUMNS))), np.zeros((0, len(ATTR_COLUMNS))), 0.0
    return np.concatenate(ys), np.concatenate(ps), total_loss / max(total, 1)


def _best_thresholds(y_true, y_prob):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_t, best_f1 = CLASSIFIER_CONFIG["threshold"], -1.0
        for t in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(y_true[:, i], (y_prob[:, i] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        thresholds.append(best_t)
    return thresholds


def classifier_metrics(y_true, y_prob, thresholds):
    """Compute per-attribute accuracy, F1, precision, recall, AUC."""
    out = {}
    for i, col in enumerate(ATTR_COLUMNS):
        key = col.lower()
        pred = (y_prob[:, i] >= thresholds[i]).astype(int)
        out[f"acc_{key}"] = float((pred == y_true[:, i]).mean())
        out[f"f1_{key}"] = float(f1_score(y_true[:, i], pred, zero_division=0))
        out[f"precision_{key}"] = float(precision_score(y_true[:, i], pred, zero_division=0))
        out[f"recall_{key}"] = float(recall_score(y_true[:, i], pred, zero_division=0))
        try:
            out[f"auc_{key}"] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except Exception:
            out[f"auc_{key}"] = None
        out[f"threshold_{key}"] = float(thresholds[i])
    return out


def train_classifier(run_dir: Optional[Path] = None,
                     smoke: bool = False,
                     force: bool = False) -> Path:
    """Train the attribute classifier. Returns path to best checkpoint."""
    run_dir = Path(run_dir or PATHS["run_dir"])
    ckpt_dir = run_dir / "checkpoints" / "attr_classifier"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    if best_path.exists() and not force:
        print("[SKIP] existing classifier:", best_path)
        return best_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_dir = PATHS["manifest_dir"]
    train_recs = load_jsonl(manifest_dir / ("train_smoke.jsonl" if smoke else "train.jsonl"))
    val_recs = load_jsonl(manifest_dir / ("val_smoke.jsonl" if smoke else "val.jsonl"))

    cfg = CLASSIFIER_CONFIG
    train_ds = FaceAttrDataset(train_recs, image_size=cfg["image_size"], train=True)
    val_ds = FaceAttrDataset(val_recs, image_size=cfg["image_size"], train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"],
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"],
                            pin_memory=torch.cuda.is_available())

    model = build_attr_classifier(cfg["use_pretrained_imagenet"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss(pos_weight=_compute_pos_weight(train_recs, device))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    epochs = 2 if smoke else cfg["epochs"]
    best_f1 = -1.0
    best_thresholds_val = [cfg["threshold"]] * len(ATTR_COLUMNS)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = bce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * y.shape[0]
            seen += y.shape[0]

        y_true, y_prob, val_loss = _collect_outputs(model, val_loader, device)
        thresholds = _best_thresholds(y_true, y_prob)
        metrics = classifier_metrics(y_true, y_prob, thresholds)
        mean_f1 = float(np.mean([metrics[f"f1_{c.lower()}"] for c in ATTR_COLUMNS]))

        row = {"epoch": epoch, "train_loss": train_loss / max(seen, 1),
               "val_loss": val_loss, "mean_val_f1": mean_f1, **metrics}
        history.append(row)
        print(f"  epoch {epoch}: train_loss={row['train_loss']:.4f}  "
              f"val_loss={val_loss:.4f}  mean_f1={mean_f1:.4f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thresholds_val = thresholds
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "attr_columns": ATTR_COLUMNS,
                "thresholds": best_thresholds_val,
            }, best_path)

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(log_dir / "attr_classifier_history.csv", index=False)
    print(f"[DONE] Best classifier saved to {best_path} (F1={best_f1:.4f})")
    return best_path


# ===== Inference helper =====================================================

def load_classifier(path: Path):
    """Load trained classifier. Returns (model, device, thresholds)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_attr_classifier(False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    thresholds = ckpt.get("thresholds",
                          [CLASSIFIER_CONFIG["threshold"]] * len(ATTR_COLUMNS))
    return model, device, thresholds


@torch.no_grad()
def predict_attrs(model, device, img: Image.Image) -> np.ndarray:
    """Return attribute probabilities [Eyeglasses, Smiling, Young]."""
    tf = transforms.Compose([
        transforms.Resize((CLASSIFIER_CONFIG["image_size"],
                            CLASSIFIER_CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    return torch.sigmoid(model(x))[0].cpu().numpy()
