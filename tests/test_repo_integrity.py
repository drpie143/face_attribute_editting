import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.import_edited_outputs import copy_tree_contents, safe_extract_zip
from src.config import MODELS, TASKS
from src.evaluation.metrics import background_l1, background_ssim
from src.inference.batch_edit import _attribute_scores, _score_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_model_config_is_two_model_setup():
    assert list(MODELS) == ["sdxl", "sd15"]
    assert MODELS["sdxl"]["pretrained_model_name_or_path"] == (
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
    assert MODELS["sd15"]["pretrained_model_name_or_path"] == (
        "runwayml/stable-diffusion-v1-5"
    )
    assert TASKS == ["add_eyeglasses", "make_smiling", "make_older"]


def test_readme_local_links_exist():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    targets = []
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)", readme):
        target = match.group(1) or match.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        targets.append(target)

    missing = [target for target in targets if not (PROJECT_ROOT / target).exists()]
    assert not missing


def test_public_docs_do_not_reference_removed_model():
    removed_model = "play" + "ground"
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "notebooks" / "face_attr_edit.ipynb",
        PROJECT_ROOT / "Face_Attr_Edit_Run_Inference_Eval_From_Drive.ipynb",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert removed_model not in text
        if path.suffix == ".ipynb":
            json.loads(path.read_text(encoding="utf-8"))


def test_background_metrics_identical_images_are_perfect():
    original = Image.new("RGB", (16, 16), (100, 120, 140))
    edited = Image.new("RGB", (16, 16), (100, 120, 140))
    mask = Image.new("L", (16, 16), 0)

    assert background_l1(original, edited, mask) == 0.0
    assert background_ssim(original, edited, mask) > 0.999


def test_candidate_scoring_has_neutral_fallbacks():
    score = _score_candidate({
        "background_l1": 0.0,
        "lpips": None,
        "identity_similarity": None,
    })

    assert 0.0 <= score <= 1.0

    before = np.array([0.1, 0.2, 0.8], dtype=np.float32)
    after = np.array([0.1, 0.2, 0.25], dtype=np.float32)
    older_scores = _attribute_scores("make_older", before, after)

    assert older_scores["attr_score"] == pytest.approx(0.75)
    assert older_scores["attr_delta"] == pytest.approx(0.55)


def test_import_helpers_are_safe_and_preserve_existing_files(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "dest"
    (src / "nested").mkdir(parents=True)
    (dst / "nested").mkdir(parents=True)
    (src / "nested" / "result.txt").write_text("new", encoding="utf-8")
    (dst / "nested" / "result.txt").write_text("old", encoding="utf-8")

    copy_tree_contents(src, dst, replace=False)
    assert (dst / "nested" / "result.txt").read_text(encoding="utf-8") == "old"

    copy_tree_contents(src, dst, replace=True)
    assert (dst / "nested" / "result.txt").read_text(encoding="utf-8") == "new"

    unsafe_zip = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_zip, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with zipfile.ZipFile(unsafe_zip, "r") as zf:
        with pytest.raises(RuntimeError):
            safe_extract_zip(zf, tmp_path / "extract")
