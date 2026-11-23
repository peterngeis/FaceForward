"""Learned-model features: CLAHE -> head crop -> HOG, per camera (see README "The math")."""
from __future__ import annotations

import cv2
import numpy as np

# Bump when learned_vector's output changes; old weights.npz / feature caches are then rejected.
# v2: CLAHE before feature extraction.
# v3: HOG of each camera's head region replaces raw pixels + skin-blob features
#     (the hand-tuned "classical" skin-blob mode was removed at the same time).
FEATURE_VERSION = 3

# HOG on the head crop. 64x48 px, 8 px cells -> 8x6 cells -> 7x5 blocks x 36 = 1260 per camera.
# Chosen by leave-one-session-out tests: raw pixels stayed at ~30% on unseen sessions no matter
# how many sessions were added; this HOG climbed ~8 points per added session (60% with 4).
HOG_SIZE = (64, 48)
HOG_CELL = 8
HOG_BINS = 9
# Length of learned_vector: per camera (cells_x - 1) * (cells_y - 1) blocks x 4 cells x bins.
FEATURE_DIM = 2 * (HOG_SIZE[0] // HOG_CELL - 1) * (HOG_SIZE[1] // HOG_CELL - 1) * 4 * HOG_BINS  # 2520

# Contrast Limited Adaptive Histogram Equalization on the lightness channel only.
# 4x4 tiles on a 320x180 frame ≈ 80x45 px per tile (roughly face-sized).
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def clahe(bgr: np.ndarray) -> np.ndarray:
    """Equalize local contrast on LAB lightness, leaving color (chroma) as-is.

    Ref: Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV, 1994.
    """
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([_CLAHE.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def hog(gray: np.ndarray, cell: int = HOG_CELL, bins: int = HOG_BINS) -> np.ndarray:
    """Histogram of Oriented Gradients, NumPy only.

    Per pixel: gradient direction (unsigned, 0-180°) and strength. Each cell x cell patch
    gets a `bins`-bin histogram of directions weighted by strength (votes split between the
    two nearest bins). Each 2x2 group of cells is L2-normalized, clipped at 0.2 and
    renormalized ("L2-Hys"), which cancels brightness/contrast: the vector describes edge
    *shape* (eyes, nose, jaw, hairline orientation), not how bright or what color things are.

    Ref: Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005;
    L2-Hys clipping from Lowe, "Distinctive Image Features...", IJCV 2004. (README: References)
    """
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    pos = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / bins)
    b0 = np.floor(pos).astype(np.int64)
    frac = pos - b0
    b0 %= bins
    b1 = (b0 + 1) % bins

    ch, cw = g.shape[0] // cell, g.shape[1] // cell
    mag, b0, b1, frac = (a[: ch * cell, : cw * cell] for a in (mag, b0, b1, frac))
    cell_idx = (np.arange(ch * cell) // cell)[:, None] * cw + (np.arange(cw * cell) // cell)[None, :]
    n = ch * cw * bins
    hist = (np.bincount((cell_idx * bins + b0).ravel(), (mag * (1.0 - frac)).ravel(), n)
            + np.bincount((cell_idx * bins + b1).ravel(), (mag * frac).ravel(), n)).reshape(ch, cw, bins)

    blocks = np.concatenate([hist[:-1, :-1], hist[1:, :-1], hist[:-1, 1:], hist[1:, 1:]], axis=-1)
    blocks /= np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks /= np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-6)
    return blocks.reshape(-1).astype(np.float32)


def head_crop(bgr: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Grayscale crop of the region the head occupies, resized to HOG_SIZE.

    roi = (x0, x1, y0, y1) as fractions of the frame (config.yaml left.roi / right.roi).
    Cropping keeps HOG on the face instead of the room, which differs every session.
    """
    h, w = bgr.shape[:2]
    x0, x1, y0, y1 = roi
    crop = bgr[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_LINEAR)


def learned_vector(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_roi: tuple[float, float, float, float],
    right_roi: tuple[float, float, float, float],
) -> np.ndarray:
    """The learned model's input. train.py and run.py both call this, so they always match.

    Frames should already be at the inference size. CLAHE -> head crop -> HOG, per camera.
    """
    left = hog(head_crop(clahe(left_bgr), left_roi))
    right = hog(head_crop(clahe(right_bgr), right_roi))
    return np.concatenate([left, right])
