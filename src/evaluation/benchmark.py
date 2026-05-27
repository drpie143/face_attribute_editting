"""
Cross-model benchmark utilities for the final SDXL baseline and SD1.5 adapter.

Reads result CSVs and produces:
  - Unified comparison table (Markdown + CSV)
  - Per-metric bar charts
  - Radar chart per model
  - Summary ranking
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS, PATHS, TASKS


MODEL_META = {
    "sdxl": {"label": "SDXL", "color": "#DD8452"},
    "sd15": {"label": "SD 1.5", "color": "#4C72B0"},
}

ALL_MODELS = {
    model_id: MODEL_META.get(model_id, {"label": model_id, "color": "#666666"})
    for model_id in MODELS
}

METRIC_COLS = [
    "attr_success_rate",
    "attr_direction_success_rate",
    "lpips_mean",
    "background_l1_mean",
    "background_ssim_mean",
    "attr_score_mean",
    "attr_delta_mean",
]

METRIC_LABELS = {
    "attr_success_rate": "Attr Success Rate (higher)",
    "attr_direction_success_rate": "Attr Direction Success (higher)",
    "lpips_mean": "LPIPS (lower)",
    "background_l1_mean": "Background L1 (lower)",
    "background_ssim_mean": "Background SSIM (higher)",
    "attr_score_mean": "Attribute Score (higher)",
    "attr_delta_mean": "Attribute Delta (higher)",
}

HIGHER_IS_BETTER = {
    "attr_success_rate",
    "attr_direction_success_rate",
    "background_ssim_mean",
    "attr_score_mean",
    "attr_delta_mean",
}


def load_results_summary(csv_path: Path) -> pd.DataFrame:
    """Load a results_summary.csv, returning an empty DataFrame if missing."""
    csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def collect_all_results(
    exports_dirs: Optional[Dict[str, Path]] = None,
    results_csvs: Optional[Dict[str, Path]] = None,
) -> pd.DataFrame:
    """
    Merge results from multiple model runs into a single DataFrame.

    Accepts either a dict of export dirs containing results_summary.csv or
    explicit CSV paths. Models without results get placeholder rows.
    """
    frames: list[pd.DataFrame] = []

    for model_id in ALL_MODELS:
        df = pd.DataFrame()
        if results_csvs and model_id in results_csvs:
            df = load_results_summary(results_csvs[model_id])
        elif exports_dirs and model_id in exports_dirs:
            df = load_results_summary(Path(exports_dirs[model_id]) / "results_summary.csv")

        if len(df) > 0:
            df = df.copy()
            if "model_id" in df.columns:
                df = df[df["model_id"] == model_id].copy()
            else:
                df["model_id"] = model_id

            if len(df) > 0:
                frames.append(df)
                continue

        for task in TASKS:
            frames.append(pd.DataFrame([{
                "model_id": model_id,
                "engine": "inpainting",
                "task": task,
                "n": 0,
                **{col: np.nan for col in METRIC_COLS},
            }]))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy model-by-task comparison table."""
    if df.empty:
        return df

    rows = []
    for task in TASKS:
        task_df = df[df["task"] == task]
        for model_id in ALL_MODELS:
            model_df = task_df[task_df["model_id"] == model_id]
            label = ALL_MODELS[model_id]["label"]
            if len(model_df) == 0 or model_df.iloc[0].get("n", 0) == 0:
                row = {"Task": task, "Model": label, "N": "-"}
                row.update({METRIC_LABELS.get(c, c): "pending" for c in METRIC_COLS})
            else:
                r = model_df.iloc[0]
                row = {"Task": task, "Model": label, "N": int(r.get("n", 0))}
                for col in METRIC_COLS:
                    val = r.get(col, np.nan)
                    row[METRIC_LABELS.get(col, col)] = (
                        f"{val:.4f}" if pd.notna(val) else "N/A"
                    )
            rows.append(row)

    return pd.DataFrame(rows)


def comparison_to_markdown(comp_df: pd.DataFrame) -> str:
    """Convert a comparison DataFrame to a Markdown table string."""
    return comp_df.to_markdown(index=False)


def plot_metric_bars(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create one bar chart per metric, grouped by task with bars per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if df.empty:
        return paths

    for metric in METRIC_COLS:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(TASKS))
        width = 0.8 / max(len(ALL_MODELS), 1)

        for idx, (model_id, meta) in enumerate(ALL_MODELS.items()):
            vals = []
            for task in TASKS:
                subset = df[(df["model_id"] == model_id) & (df["task"] == task)]
                vals.append(subset[metric].values[0] if len(subset) > 0 else np.nan)
            offset = (idx - (len(ALL_MODELS) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width * 0.9, label=meta["label"],
                   color=meta["color"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", " ").title() for t in TASKS])
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = output_dir / f"benchmark_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_radar_chart(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Create a normalized radar chart showing each model's average metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    usable_metrics = [
        c for c in METRIC_COLS
        if c in df.columns and df[c].notna().any()
    ]
    if not usable_metrics:
        return None

    model_avgs: dict[str, list[float]] = {}
    for model_id in ALL_MODELS:
        sub = df[df["model_id"] == model_id]
        avgs = []
        for metric in usable_metrics:
            val = sub[metric].mean()
            if metric not in HIGHER_IS_BETTER and pd.notna(val):
                val = 1.0 - min(val, 1.0)
            avgs.append(float(val) if pd.notna(val) else 0.0)
        model_avgs[model_id] = avgs

    labels = [METRIC_LABELS.get(metric, metric).split("(")[0].strip()
              for metric in usable_metrics]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model_id, meta in ALL_MODELS.items():
        vals = model_avgs[model_id] + model_avgs[model_id][:1]
        ax.plot(angles, vals, "o-", label=meta["label"],
                color=meta["color"], linewidth=2)
        ax.fill(angles, vals, alpha=0.15, color=meta["color"])

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (normalized, higher is better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    path = output_dir / "benchmark_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by average normalized score across available metrics."""
    if df.empty:
        return df

    scores = []
    for model_id in ALL_MODELS:
        sub = df[df["model_id"] == model_id]
        if sub.empty or sub["n"].sum() == 0:
            scores.append({
                "Model": ALL_MODELS[model_id]["label"],
                "Status": "pending",
                "Overall Score": np.nan,
            })
            continue

        total = 0.0
        count = 0
        for metric in METRIC_COLS:
            if metric not in sub.columns:
                continue
            val = sub[metric].mean()
            if pd.isna(val):
                continue
            if metric not in HIGHER_IS_BETTER:
                val = 1.0 - min(val, 1.0)
            total += float(val)
            count += 1

        scores.append({
            "Model": ALL_MODELS[model_id]["label"],
            "Status": "evaluated",
            "Overall Score": round(total / max(count, 1), 4),
        })

    ranking = pd.DataFrame(scores).sort_values(
        "Overall Score",
        ascending=False,
        na_position="last",
    )
    ranking["Rank"] = range(1, len(ranking) + 1)
    return ranking[["Rank", "Model", "Status", "Overall Score"]]


def run_benchmark(
    exports_dirs: Optional[Dict[str, Path]] = None,
    results_csvs: Optional[Dict[str, Path]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run the full benchmark and save tables, charts, and ranking.

    Returns a dict with keys: merged_df, comparison, ranking, figures.
    """
    output_dir = Path(output_dir or (PATHS["project_root"] / "results" / "benchmark"))
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = collect_all_results(exports_dirs, results_csvs)

    comp = make_comparison_table(merged)
    comp.to_csv(output_dir / "comparison_table.csv", index=False)
    (output_dir / "comparison_table.md").write_text(
        comparison_to_markdown(comp),
        encoding="utf-8",
    )
    print("[SAVED] comparison_table.csv / .md")

    fig_dir = output_dir / "figures"
    bar_paths = plot_metric_bars(merged, fig_dir)
    radar_path = plot_radar_chart(merged, fig_dir)

    ranking = compute_ranking(merged)
    ranking.to_csv(output_dir / "ranking.csv", index=False)
    print("[SAVED] ranking.csv")
    print(ranking.to_string(index=False))

    return {
        "merged_df": merged,
        "comparison": comp,
        "ranking": ranking,
        "figures": bar_paths + ([radar_path] if radar_path else []),
    }
