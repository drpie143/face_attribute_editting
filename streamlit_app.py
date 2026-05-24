import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS, NEGATIVE_PROMPTS, PATHS, TARGET_PROMPTS, TASKS


RUN_DIR = PATHS["run_dir"]
EXPORTS_DIR = PATHS["exports_dir"]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
ASSET_DIR = PROJECT_ROOT / "docs" / "assets"

TASK_LABELS = {
    "add_eyeglasses": "Add eyeglasses",
    "make_smiling": "Make smiling",
    "make_older": "Make older",
}


st.set_page_config(
    page_title="Face Attribute Editing Benchmark",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .app-title { font-size: 2.25rem; font-weight: 750; margin-bottom: 0.25rem; }
    .app-subtitle { color: #5c6670; margin-bottom: 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_title(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='app-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def resolve_project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def model_lora_path(model_id: str) -> Path:
    run_path = RUN_DIR / "loras" / model_id / "pytorch_lora_weights.safetensors"
    if run_path.exists():
        return run_path
    return EXPORTS_DIR / "best_loras" / f"{model_id}_faceattr_lora.safetensors"


def available_comparison_index(selected_only: bool) -> pd.DataFrame:
    if selected_only:
        index = read_csv(str(BENCHMARK_DIR / "selected_visual_index.csv"))
        if "image_id" in index.columns:
            index["image_id"] = index["image_id"].astype(str).str.zfill(5)
        return index
    full_index = read_csv(str(EXPORTS_DIR / "visual_index.csv"))
    if full_index.empty:
        index = read_csv(str(BENCHMARK_DIR / "selected_visual_index.csv"))
        if "image_id" in index.columns:
            index["image_id"] = index["image_id"].astype(str).str.zfill(5)
        return index
    if "asset_path" not in full_index.columns and "comparison_path" in full_index.columns:
        full_index = full_index.rename(columns={"comparison_path": "asset_path"})
    if "image_id" in full_index.columns:
        full_index["image_id"] = full_index["image_id"].astype(str).str.zfill(5)
    return full_index


def show_benchmark_dashboard() -> None:
    page_title(
        "Face Attribute Editing Benchmark",
        "Standard metrics use real edited outputs and metadata for both final models.",
    )

    raw_samples = ASSET_DIR / "raw_samples.jpg"
    if raw_samples.exists():
        st.image(str(raw_samples), caption="Raw CelebA-HQ style input samples", width="stretch")

    health = read_csv(str(BENCHMARK_DIR / "model_health_check.csv"))
    ranking = read_csv(str(BENCHMARK_DIR / "ranking.csv"))
    comparison = read_csv(str(BENCHMARK_DIR / "comparison_table.csv"))

    st.subheader("Model Status")
    model_cols = st.columns(len(MODELS))
    for col, model_id in zip(model_cols, MODELS):
        lora = model_lora_path(model_id)
        with col:
            st.metric(model_id, "LoRA found" if lora.exists() else "LoRA missing")
            st.caption(str(lora.relative_to(PROJECT_ROOT)) if lora.exists() else "No local weights detected")

    if not health.empty:
        st.dataframe(health, width="stretch")

    st.subheader("Benchmark Summary")
    if not comparison.empty:
        st.dataframe(comparison, width="stretch")
    else:
        st.warning("benchmark/comparison_table.csv was not found.")

    if not ranking.empty:
        st.subheader("Standard Evaluation Status")
        st.dataframe(ranking, width="stretch")

    st.subheader("Benchmark Charts")
    chart_files = [
        BENCHMARK_DIR / "figures" / "standard_attribute_success.png",
        BENCHMARK_DIR / "figures" / "standard_direction_success.png",
        BENCHMARK_DIR / "figures" / "standard_attribute_score.png",
        BENCHMARK_DIR / "figures" / "standard_background_ssim.png",
        BENCHMARK_DIR / "figures" / "standard_lpips.png",
    ]
    chart_cols = st.columns(2)
    shown = 0
    for idx, chart_path in enumerate(chart_files):
        if chart_path.exists():
            with chart_cols[idx % 2]:
                st.image(str(chart_path), width="stretch")
            shown += 1
    if shown == 0:
        st.info("No chart images found in benchmark/figures.")


def show_comparison_browser() -> None:
    page_title(
        "Browse Final Comparisons",
        "Inspect original, SD 1.5 best, and SDXL best panels. The README set contains exactly three images per task.",
    )

    selected_only = st.toggle("Use README-selected examples only", value=True)
    index = available_comparison_index(selected_only)
    if index.empty:
        st.warning("No visual index found. Expected benchmark/selected_visual_index.csv or exports/visual_index.csv.")
        return

    task = st.selectbox(
        "Task",
        [t for t in TASKS if t in set(index["task"])],
        format_func=lambda t: TASK_LABELS.get(t, t),
    )
    task_df = index[index["task"] == task].copy()
    image_id = st.selectbox("Image ID", sorted(task_df["image_id"].astype(str).unique()))
    row = task_df[task_df["image_id"].astype(str) == str(image_id)].iloc[0]
    image_path = resolve_project_path(row["asset_path"])

    if image_path.exists():
        st.image(str(image_path), caption=f"{TASK_LABELS.get(task, task)} - {image_id}", width="stretch")
    else:
        st.error(f"Missing comparison image: {image_path}")

    with st.expander("Selection notes"):
        st.write("Fill these fields in benchmark/selected_visual_index.csv when manually choosing the best model per sample.")
        st.dataframe(task_df, width="stretch")


def make_demo_mask(task: str, size: int) -> tuple[Image.Image, Image.Image]:
    hard = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(hard)

    if task == "add_eyeglasses":
        draw.rectangle(
            [int(size * 0.22), int(size * 0.32), int(size * 0.78), int(size * 0.55)],
            fill=255,
        )
        blur_radius = 5
    elif task == "make_smiling":
        draw.ellipse(
            [int(size * 0.30), int(size * 0.60), int(size * 0.70), int(size * 0.82)],
            fill=255,
        )
        blur_radius = 7
    else:
        draw.ellipse(
            [int(size * 0.18), int(size * 0.18), int(size * 0.82), int(size * 0.92)],
            fill=255,
        )
        draw.rectangle(
            [int(size * 0.22), int(size * 0.32), int(size * 0.78), int(size * 0.55)],
            fill=0,
        )
        draw.ellipse(
            [int(size * 0.30), int(size * 0.60), int(size * 0.70), int(size * 0.82)],
            fill=0,
        )
        blur_radius = 8

    soft = hard.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return hard, soft


def center_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def show_interactive_editor() -> None:
    page_title(
        "Interactive Editor",
        "Upload a face image and run one of the two final LoRA adapters. A simple geometric mask is used for custom images.",
    )

    uploaded = st.file_uploader("Face image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return

    source = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1, 1])

    with left:
        st.image(source, caption="Input", width="stretch")
        task = st.selectbox("Task", TASKS, format_func=lambda t: TASK_LABELS.get(t, t))
        model_id = st.selectbox("Model", list(MODELS.keys()))
        strength = st.slider("Strength", 0.10, 1.00, float(MODELS[model_id].get("demo_strength", 0.55)), 0.05)
        guidance = st.slider("Guidance scale", 1.0, 15.0, float(MODELS[model_id].get("guidance_scale", 7.5)), 0.5)
        steps = st.slider("Inference steps", 10, 50, 25, 5)
        run = st.button("Apply edit", type="primary")

    if not run:
        return

    with right:
        try:
            import torch

            from src.inference.pipeline import blend_with_soft_mask, load_pipeline, unload_pipeline

            resolution = int(MODELS[model_id]["resolution"])
            image = center_square(source).resize((resolution, resolution), Image.Resampling.LANCZOS)
            mask_hard, mask_soft = make_demo_mask(task, resolution)

            st.image(mask_soft, caption="Generated mask", width=220)
            st.caption(f"Loading {model_id} from {model_lora_path(model_id)}")

            pipe = load_pipeline(model_id, run_dir=RUN_DIR)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(42)

            with st.spinner(f"Running inference on {device}"):
                with torch.inference_mode():
                    result = pipe(
                        prompt=TARGET_PROMPTS[task],
                        negative_prompt=NEGATIVE_PROMPTS[task],
                        image=image,
                        mask_image=mask_hard,
                        strength=strength,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        generator=generator,
                    ).images[0]
                blend = blend_with_soft_mask(image, result, mask_soft)

            unload_pipeline(pipe)
            st.image(blend, caption=f"{model_id} result", width="stretch")
        except Exception as exc:
            st.error(f"Interactive inference failed: {exc}")
            st.info("Check that Diffusers dependencies, CUDA, and the local LoRA weights are available.")


page = st.sidebar.radio(
    "Page",
    ["Benchmark Dashboard", "Browse Final Comparisons", "Interactive Editor"],
)

if page == "Benchmark Dashboard":
    show_benchmark_dashboard()
elif page == "Browse Final Comparisons":
    show_comparison_browser()
else:
    show_interactive_editor()
