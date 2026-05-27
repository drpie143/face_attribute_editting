# Face Attribute Editing

Mask-guided face attribute editing with LoRA-finetuned diffusion models.

![Raw CelebA-HQ style samples](docs/assets/raw_samples.jpg)

## Overview

Given an input face image, a target edit, and a semantic task mask, this project
generates an edited image that changes the requested attribute while preserving
identity, pose, background, and unrelated facial regions.

Final edit tasks:

- `add_eyeglasses`
- `make_smiling`
- `make_older`

Final model comparison:

- **Baseline:** SDXL LoRA (`sdxl`) built on `stabilityai/stable-diffusion-xl-base-1.0`
- **Comparison model:** Stable Diffusion 1.5 LoRA (`sd15`) built on `runwayml/stable-diffusion-v1-5`

The public benchmark and demo use only these two final adapters.

## Selected Results

Each README panel is arranged as:

```text
original | sd15 best | sdxl best
```

The full generated output set stays local in `exports/` and is ignored by git.

### Add Eyeglasses

![add eyeglasses 04181](docs/assets/examples/add_eyeglasses_04181_manual.jpg)

![add eyeglasses 12340](docs/assets/examples/add_eyeglasses_12340_manual.jpg)

![add eyeglasses 03108](docs/assets/examples/add_eyeglasses_03108_manual.jpg)

### Make Smiling

![make smiling 02457](docs/assets/examples/make_smiling_02457_manual.jpg)

![make smiling 05199](docs/assets/examples/make_smiling_05199_manual.jpg)

![make smiling 22964](docs/assets/examples/make_smiling_22964_manual.jpg)

### Make Older

![make older 00742](docs/assets/examples/make_older_00742_manual.jpg)

![make older 21163](docs/assets/examples/make_older_21163_manual.jpg)

![make older 20842](docs/assets/examples/make_older_20842_manual.jpg)

## Benchmark

The standard benchmark is computed from real edited outputs and metadata:

```text
original image + task mask + selected edited image + candidate metadata
```

README comparison panels are for visual inspection only. Panel-level pixel
metrics are kept as a lightweight sanity check, not as the main benchmark.

### Metrics

- **Hard attribute success:** classifier output crosses the task threshold.
- **Direction success:** classifier output moves in the requested direction by at least `0.05`.
- **Attribute score:** task-aware soft score; for `make_older`, this is `1 - P(Young)`.
- **Attribute delta:** task-aware improvement from original to edited image.
- **LPIPS:** perceptual distance from the source image. Lower is better.
- **Background L1 / Background SSIM:** preservation outside the edit mask.
- **Manual preference:** recommended for final qualitative review because face editing quality is partly subjective.

Candidate reranking uses:

```text
attr_score=0.70, attr_delta=0.15, background_score=0.10, lpips_score=0.05
```

![standard attribute success](benchmark/figures/standard_attribute_success.png)

![standard direction success](benchmark/figures/standard_direction_success.png)

![standard attribute score](benchmark/figures/standard_attribute_score.png)

![standard background ssim](benchmark/figures/standard_background_ssim.png)

![standard lpips](benchmark/figures/standard_lpips.png)

| Task | Model | N | Hard success | Direction success | Attr score | Attr delta | LPIPS | Background SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `add_eyeglasses` | SDXL | 12 | 0.4167 | 0.4167 | 0.1610 | 0.1487 | 0.0452 | 0.9965 |
| `add_eyeglasses` | SD 1.5 | 12 | 0.8333 | 0.9167 | 0.7452 | 0.7333 | 0.0783 | 0.9936 |
| `make_smiling` | SDXL | 12 | 0.3333 | 0.5833 | 0.3538 | 0.2967 | 0.0160 | 0.9967 |
| `make_smiling` | SD 1.5 | 12 | 0.0833 | 0.3333 | 0.1479 | 0.0888 | 0.0181 | 0.9961 |
| `make_older` | SDXL | 12 | 0.0000 | 0.1667 | 0.1132 | 0.0256 | 0.0220 | 0.9909 |
| `make_older` | SD 1.5 | 12 | 0.0000 | 0.5000 | 0.2937 | 0.1732 | 0.0448 | 0.9861 |

`make_older` has zero hard-threshold success for both models. Direction and
soft-score metrics show partial movement on some samples, but none cross the
classifier's `Young` threshold. Treat this as a classifier-threshold result, not
as a final human-preference result.

Tracked benchmark files:

- [`benchmark/comparison_table.csv`](benchmark/comparison_table.csv)
- [`benchmark/comparison_table.md`](benchmark/comparison_table.md)
- [`benchmark/ranking.csv`](benchmark/ranking.csv)
- [`benchmark/final_model_benchmark.csv`](benchmark/final_model_benchmark.csv)
- [`benchmark/panel_preservation_summary.csv`](benchmark/panel_preservation_summary.csv)
- [`benchmark/panel_preservation_long.csv`](benchmark/panel_preservation_long.csv)
- [`benchmark/model_health_check.csv`](benchmark/model_health_check.csv)
- [`benchmark/selected_visual_index.csv`](benchmark/selected_visual_index.csv)

## Repository Structure

```text
configs/       Project configuration
src/           Data, training, inference, evaluation, and reporting modules
scripts/       CLI entry points
notebooks/     Notebook entry points
docs/assets/   Lightweight README images selected from final outputs
benchmark/     Lightweight benchmark CSVs and charts for GitHub
data/          Local raw/processed data, ignored by git
runs/          Local training and inference outputs, ignored by git
exports/       Full final artifacts, ignored by git
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

For CUDA environments, install the PyTorch build that matches your driver first
if the default `pip` wheel is not appropriate.

## Reproduce

Preprocess data and create manifests:

```bash
python scripts/run_preprocessing.py --download-if-missing --max-images 2000
```

Train the attribute classifier and the two final LoRA adapters:

```bash
python scripts/run_training.py --classifier
python scripts/run_training.py --model-id sd15
python scripts/run_training.py --model-id sdxl
```

Run inference and evaluation:

```bash
python scripts/run_inference.py --model-id sd15
python scripts/run_inference.py --model-id sdxl
python scripts/run_evaluation.py
python scripts/run_benchmark.py
```

If edited-output archives already exist, import and rerank them instead:

```bash
python scripts/import_edited_outputs.py --source sd15-20260524T033232Z-3-001.zip --model-id sd15 --replace-model-dir
python scripts/import_edited_outputs.py --source sdxl-20260524T033538Z-3-001.zip --model-id sdxl --replace-model-dir
python scripts/rerank_candidates.py --model-id sd15 --model-id sdxl
python scripts/run_evaluation.py
```

For Drive/GPU inference and final visual generation, use:

```text
Face_Attr_Edit_Run_Inference_Eval_From_Drive.ipynb
```

## Demo

Run the Streamlit demo:

```bash
streamlit run streamlit_app.py
```

The demo provides:

- benchmark dashboard
- final comparison browser
- interactive editor for local LoRA weights

## Notes

- Full model weights are not tracked in git.
- Expected local weights are under `runs/face_attr_edit_v2_full_inpaint/loras/`
  or `exports/best_loras/`.
- `data/`, `runs/`, `exports/`, zips, checkpoints, and model weights are ignored.
- Public assets are intentionally limited to README examples and lightweight benchmark files.

## License

MIT. See [LICENSE](LICENSE).
