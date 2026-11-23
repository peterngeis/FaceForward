"""Training-time augmentation: synthetic lighting and head-position variation.

Applied to frames already resized to the inference size, before feature extraction.
Each camera gets independent random parameters (the two cameras auto-expose and sit
at different angles, so their lighting and framing never change in lockstep).

Refs: Simard et al., ICDAR 2003 (affine distortions); Shorten & Khoshgoftaar,
J. Big Data 2019 (augmentation survey). (README: References)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import AugmentSpec


def position(bgr: np.ndarray, rng: np.random.Generator, spec: AugmentSpec) -> np.ndarray:
    """Random scale / rotation / shift: simulates slouching, leaning, seat height, distance."""
    h, w = bgr.shape[:2]
    s = rng.uniform(1.0 - spec.scale, 1.0 + spec.scale)
    a = rng.uniform(-spec.rotate_deg, spec.rotate_deg)
    m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), a, s)
    m[0, 2] += rng.uniform(-spec.shift, spec.shift) * w
    m[1, 2] += rng.uniform(-spec.shift, spec.shift) * h
    # Replicate (not reflect) borders so zoom-out never mirrors a second face into frame.
    return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def lighting(bgr: np.ndarray, rng: np.random.Generator, spec: AugmentSpec) -> np.ndarray:
    """Random white balance, exposure gain, contrast and gamma: simulates morning vs night."""
    x = bgr.astype(np.float32) / 255.0
    t = rng.uniform(-spec.white_balance, spec.white_balance)  # >0 warm (lamp), <0 cool (daylight)
    x[:, :, 0] *= 1.0 - t
    x[:, :, 2] *= 1.0 + t
    x *= rng.uniform(1.0 - spec.gain, 1.0 + spec.gain)
    c = rng.uniform(1.0 - spec.contrast, 1.0 + spec.contrast)
    mean = float(x.mean())
    x = (x - mean) * c + mean
    # Log-uniform gamma so brightening (<1) and darkening (>1) are equally likely.
    g = math.exp(rng.uniform(math.log(spec.gamma[0]), math.log(spec.gamma[1])))
    x = np.clip(x, 0.0, 1.0) ** g
    return (x * 255.0 + 0.5).astype(np.uint8)


def augment_pair(
    left: np.ndarray, right: np.ndarray, rng: np.random.Generator, spec: AugmentSpec
) -> tuple[np.ndarray, np.ndarray]:
    return (
        lighting(position(left, rng, spec), rng, spec),
        lighting(position(right, rng, spec), rng, spec),
    )
