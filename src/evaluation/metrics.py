"""
Image-quality and editing-quality metrics.

  - LPIPS (perceptual distance)
  - Background L1 (pixel-level background preservation)
  - Background SSIM (structural similarity on background)
  - Identity similarity (InsightFace cosine, optional)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


# ===== LPIPS ================================================================

_LPIPS_MODEL = None


def _get_lpips_model():
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(dev)
        _LPIPS_MODEL.eval()
    except Exception as e:
        print(f"[WARN] LPIPS unavailable: {e}")
        _LPIPS_MODEL = None
    return _LPIPS_MODEL


def lpips_distance(a_img: Image.Image, b_img: Image.Image) -> Optional[float]:
    """Compute LPIPS perceptual distance between two images."""
    model = _get_lpips_model()
    if model is None:
        return None
    a = np.array(a_img.convert("RGB").resize((256, 256))).astype(np.float32) / 127.5 - 1
    b = np.array(b_img.convert("RGB").resize((256, 256))).astype(np.float32) / 127.5 - 1
    ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0)
    dev = next(model.parameters()).device
    ta, tb = ta.to(dev), tb.to(dev)
    with torch.no_grad():
        return float(model(ta, tb).item())


# ===== Background L1 ========================================================

def background_l1(original: Image.Image, edited: Image.Image,
                  mask: Image.Image) -> float:
    """Mean absolute pixel difference in the background (non-masked) region."""
    o = np.asarray(original.convert("RGB")).astype(np.float32) / 255.0
    e = np.asarray(
        edited.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    ).astype(np.float32) / 255.0
    m = np.asarray(
        mask.convert("L").resize(original.size, Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0
    bg = 1.0 - m[..., None]
    return float((np.abs(o - e) * bg).sum() / max(float(bg.sum()) * 3.0, 1e-6))


# ===== Background SSIM ======================================================

def background_ssim(original: Image.Image, edited: Image.Image,
                    mask: Image.Image) -> float:
    """Structural similarity on the background region."""
    o = np.asarray(original.convert("L")).astype(np.float32) / 255.0
    e = np.asarray(
        edited.convert("L").resize(original.size, Image.Resampling.LANCZOS)
    ).astype(np.float32) / 255.0
    m = np.asarray(
        mask.convert("L").resize(original.size, Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0
    bg = 1.0 - m

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx = cv2.GaussianBlur(o, (11, 11), 1.5)
    my = cv2.GaussianBlur(e, (11, 11), 1.5)
    sx = cv2.GaussianBlur(o * o, (11, 11), 1.5) - mx * mx
    sy = cv2.GaussianBlur(e * e, (11, 11), 1.5) - my * my
    sxy = cv2.GaussianBlur(o * e, (11, 11), 1.5) - mx * my
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / (
        (mx * mx + my * my + c1) * (sx + sy + c2) + 1e-8)
    return float((ssim * bg).sum() / max(float(bg.sum()), 1e-6))


# ===== Identity similarity (InsightFace) ====================================

_INSIGHTFACE_APP = None
_INSIGHTFACE_TRIED = False


def _get_insightface_app():
    global _INSIGHTFACE_APP, _INSIGHTFACE_TRIED
    if _INSIGHTFACE_TRIED:
        return _INSIGHTFACE_APP
    _INSIGHTFACE_TRIED = True
    try:
        from insightface.app import FaceAnalysis
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if torch.cuda.is_available()
                     else ["CPUExecutionProvider"])
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx = 0 if torch.cuda.is_available() else -1
        app.prepare(ctx_id=ctx, det_size=(640, 640))
        _INSIGHTFACE_APP = app
        print("[OK] InsightFace available")
    except Exception as e:
        print(f"[WARN] InsightFace unavailable: {e}")
        _INSIGHTFACE_APP = None
    return _INSIGHTFACE_APP


def identity_similarity(original: Image.Image,
                        edited: Image.Image) -> Optional[float]:
    """Cosine similarity of face embeddings (InsightFace). None if unavailable."""
    app = _get_insightface_app()
    if app is None:
        return None
    try:
        fa = app.get(cv2.cvtColor(np.array(original.convert("RGB")), cv2.COLOR_RGB2BGR))
        fb = app.get(cv2.cvtColor(np.array(edited.convert("RGB")), cv2.COLOR_RGB2BGR))
        if not fa or not fb:
            return None
        a = fa[0].embedding.astype(np.float32)
        b = fb[0].embedding.astype(np.float32)
        return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6))
    except Exception:
        return None
