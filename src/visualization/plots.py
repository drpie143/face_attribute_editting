"""
Training curve plots and metric visualisations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_classifier_history(log_dir: Path,
                            output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Plot the attribute classifier training curves (loss + F1).
    """
    csv_path = log_dir / "attr_classifier_history.csv"
    if not csv_path.exists():
        print("[WARN] no classifier history:", csv_path)
        return None

    df = pd.read_csv(csv_path)
    output_dir = output_dir or log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")

    if "train_loss" in df.columns:
        ax1.plot(df["epoch"], df["train_loss"], "o-",
                 color="tab:red", alpha=0.7, label="Train Loss")
    if "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["val_loss"], "s-",
                 color="tab:orange", alpha=0.7, label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")

    if "mean_val_f1" in df.columns:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Mean Val F1", color="tab:blue")
        ax2.plot(df["epoch"], df["mean_val_f1"], "^-",
                 color="tab:blue", linewidth=2, label="Mean Val F1")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.legend(loc="upper right")

    plt.title("Attribute Classifier Training")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "classifier_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path


def plot_training_summary(status_csv: Path,
                          output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Bar chart showing training time and steps for each model.
    """
    if not status_csv.exists():
        return None

    df = pd.read_csv(status_csv)
    output_dir = output_dir or status_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    models = df["model_id"].tolist()

    ax1.barh(models, df["steps"], color=colors[:len(models)])
    ax1.set_xlabel("Training Steps")
    ax1.set_title("Training Steps per Model")

    ax2.barh(models, df["elapsed_min"], color=colors[:len(models)])
    ax2.set_xlabel("Time (minutes)")
    ax2.set_title("Training Time per Model")

    plt.tight_layout()
    path = output_dir / "training_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
    return path
