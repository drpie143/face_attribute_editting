"""
Cross-model benchmark: compare SD1.5, SDXL, and Playground v2.5 side-by-side.

Reads results CSVs from each model's exports and produces:
  - Unified comparison table (Markdown + CSV)
  - Per-metric bar charts
  - Radar (spider) chart per model
  - Summary ranking
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS, TASKS


# ===== Model registry for benchmark ========================================

ALL_MODELS = {
    "sd15": {"label": "SD 1.5", "color": "#4C72B0"},
    "sdxl": {"label": "SDXL", "color": "#DD8452"},
    "playground25": {"label": "Playground v2.5", "color": "#55A868"},
}

METRIC_COLS = [
    "attr_success_rate",
    "lpips_mean",
    "background_l1_mean",
    "background_ssim_mean",
    "identity_similarity_mean",
    "selection_score_mean",
]

METRIC_LABELS = {
    "attr_success_rate": "Attr Success Rate ↑",
    "lpips_mean": "LPIPS ↓",
    "background_l1_mean": "Background L1 ↓",
    "background_ssim_mean": "Background SSIM ↑",
    "identity_similarity_mean": "Identity Sim ↑",
    "selection_score_mean": "Selection Score ↑",
}

# Higher is better for these; lower is better for the rest
HIGHER_IS_BETTER = {"attr_success_rate", "background_ssim_mean",
                    "identity_similarity_mean", "selection_score_mean"}


# ===== Data loading =========================================================

def load_results_summary(csv_path: Path) -> pd.DataFrame:
    """Load a results_summary.csv, returning an empty DF if missing."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def collect_all_results(
    exports_dirs: Optional[Dict[str, Path]] = None,
    results_csvs: Optional[Dict[str, Path]] = None,
) -> pd.DataFrame:
    """
    Merge results from multiple model runs into a single DataFrame.

    Accepts either a dict of exports dirs (each containing results_summary.csv)
    or explicit CSV paths. Models without results get NaN rows.
    """
    frames = []

    for model_id in ALL_MODELS:
        df = pd.DataFrame()
        if results_csvs and model_id in results_csvs:
            df = load_results_summary(results_csvs[model_id])
        elif exports_dirs and model_id in exports_dirs:
            csv = exports_dirs[model_id] / "results_summary.csv"
            df = load_results_summary(csv)

        if len(df) > 0:
            if "model_id" in df.columns:
                df_filtered = df[df["model_id"] == model_id].copy()
                if len(df_filtered) > 0:
                    frames.append(df_filtered)
                else:
                    # If this model is not in the dataframe, treat as missing/placeholder
                    for task in TASKS:
                        frames.append(pd.DataFrame([{
                            "model_id": model_id,
                            "engine": "inpainting",
                            "task": task,
                            "n": 0,
                            **{col: np.nan for col in METRIC_COLS},
                        }]))
            else:
                df = df.copy()
                df["model_id"] = model_id
                frames.append(df)
        else:
            # Placeholder rows for models without results yet
            for task in TASKS:
                frames.append(pd.DataFrame([{
                    "model_id": model_id,
                    "engine": "inpainting",
                    "task": task,
                    "n": 0,
                    **{col: np.nan for col in METRIC_COLS},
                }]))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



# ===== Comparison table =====================================================

def make_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the results into a model × task comparison table.
    Returns a tidy DataFrame suitable for display.
    """
    if df.empty:
        return df

    rows = []
    for task in TASKS:
        task_df = df[df["task"] == task]
        for model_id in ALL_MODELS:
            model_df = task_df[task_df["model_id"] == model_id]
            label = ALL_MODELS[model_id]["label"]
            if len(model_df) == 0 or model_df.iloc[0].get("n", 0) == 0:
                row = {"Task": task, "Model": label, "N": "—"}
                row.update({METRIC_LABELS.get(c, c): "pending" for c in METRIC_COLS})
            else:
                r = model_df.iloc[0]
                row = {"Task": task, "Model": label, "N": int(r.get("n", 0))}
                for c in METRIC_COLS:
                    val = r.get(c, np.nan)
                    row[METRIC_LABELS.get(c, c)] = (
                        f"{val:.4f}" if pd.notna(val) else "N/A")
            rows.append(row)

    return pd.DataFrame(rows)


def comparison_to_markdown(comp_df: pd.DataFrame) -> str:
    """Convert comparison DataFrame to a Markdown table string."""
    return comp_df.to_markdown(index=False)


# ===== Visualisation ========================================================

def plot_metric_bars(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create one bar chart per metric, grouped by task with bars per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for metric in METRIC_COLS:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(TASKS))
        width = 0.25
        offsets = np.linspace(-width, width, len(ALL_MODELS))

        for i, (model_id, meta) in enumerate(ALL_MODELS.items()):
            vals = []
            for task in TASKS:
                subset = df[(df["model_id"] == model_id) & (df["task"] == task)]
                v = subset[metric].values[0] if len(subset) > 0 else np.nan
                vals.append(v)
            ax.bar(x + offsets[i], vals, width * 0.9,
                   label=meta["label"], color=meta["color"], alpha=0.85)

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
    """
    Spider / radar chart showing each model's average across metrics.
    Metrics are normalised to [0, 1] for comparability.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    usable_metrics = [c for c in METRIC_COLS if c in df.columns and df[c].notna().any()]
    if not usable_metrics:
        return None

    # Per-model average across tasks
    model_avgs = {}
    for model_id in ALL_MODELS:
        sub = df[df["model_id"] == model_id]
        avgs = []
        for m in usable_metrics:
            v = sub[m].mean()
            # Flip sign for "lower is better" metrics so higher = better on chart
            if m not in HIGHER_IS_BETTER and pd.notna(v):
                v = 1.0 - min(v, 1.0)
            avgs.append(v if pd.notna(v) else 0.0)
        model_avgs[model_id] = avgs

    # Radar plot
    labels = [METRIC_LABELS.get(m, m).split("↑")[0].split("↓")[0].strip()
              for m in usable_metrics]
    num = len(labels)
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model_id, meta in ALL_MODELS.items():
        vals = model_avgs[model_id] + model_avgs[model_id][:1]
        ax.plot(angles, vals, "o-", label=meta["label"],
                color=meta["color"], linewidth=2)
        ax.fill(angles, vals, alpha=0.15, color=meta["color"])
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison (normalised, higher = better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    path = output_dir / "benchmark_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ===== Ranking ==============================================================

def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models by an overall score averaged across tasks and metrics.
    """
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
        for m in METRIC_COLS:
            if m not in sub.columns:
                continue
            v = sub[m].mean()
            if pd.isna(v):
                continue
            if m not in HIGHER_IS_BETTER:
                v = 1.0 - min(v, 1.0)
            total += v
            count += 1

        scores.append({
            "Model": ALL_MODELS[model_id]["label"],
            "Status": "evaluated",
            "Overall Score": round(total / max(count, 1), 4),
        })

    ranking = pd.DataFrame(scores).sort_values("Overall Score",
                                                ascending=False, na_position="last")
    ranking["Rank"] = range(1, len(ranking) + 1)
    return ranking[["Rank", "Model", "Status", "Overall Score"]]


# ===== Main benchmark entry point ===========================================

def run_benchmark(
    exports_dirs: Optional[Dict[str, Path]] = None,
    results_csvs: Optional[Dict[str, Path]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run the full benchmark: collect results, generate tables, charts, ranking.

    Returns a dict with keys: 'merged_df', 'comparison', 'ranking', 'figures'.
    """
    output_dir = Path(output_dir or (PATHS["project_root"] / "results" / "benchmark"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect
    merged = collect_all_results(exports_dirs, results_csvs)

    # 2. Comparison table
    comp = make_comparison_table(merged)
    comp.to_csv(output_dir / "comparison_table.csv", index=False)
    md = comparison_to_markdown(comp)
    (output_dir / "comparison_table.md").write_text(md, encoding="utf-8")
    print("[SAVED] comparison_table.csv / .md")

    # 3. Charts
    fig_dir = output_dir / "figures"
    bar_paths = plot_metric_bars(merged, fig_dir)
    radar_path = plot_radar_chart(merged, fig_dir)

    # 4. Ranking
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


# Allow importing PATHS here for the main entry point
from src.config import PATHS  # noqa: E402 (already imported above, redeclared for clarity)
