from __future__ import annotations

import cv2
import numpy as np

from .hysteresis import SwitchSmoother


def draw_roi(img: np.ndarray, roi: tuple[float, float, float, float], color=(255, 200, 0)) -> None:
    """Outline the head region the learned model looks at (in place). Keep your head inside it."""
    h, w = img.shape[:2]
    x0, x1, y0, y1 = roi
    cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), color, 1, cv2.LINE_AA)


def draw_hud(
    left: np.ndarray,
    right: np.ndarray,
    smoother: SwitchSmoother,
    s_left: float,
    s_right: float,
    locked: bool,
    rois: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    h, w = 240, 320
    l = cv2.resize(left, (w, h))
    r = cv2.resize(right, (w, h))
    if rois is not None:
        draw_roi(l, rois[0])
        draw_roi(r, rois[1])
    canvas = np.zeros((h + 80, w * 2 + 16, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = l
    canvas[0:h, w + 16 :] = r

    sel = smoother.selected
    for name, x0, score in (("LEFT", 0, s_left), ("RIGHT", w + 16, s_right)):
        on = (name == "LEFT" and sel == "left") or (name == "RIGHT" and sel == "right")
        color = (40, 190, 255) if on else (90, 90, 90)
        cv2.rectangle(canvas, (x0, 0), (x0 + w - 1, h - 1), color, 2 if on else 1)
        cv2.putText(canvas, f"{name}  {score:.2f}", (x0 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    y = h + 28
    cv2.putText(
        canvas,
        f"out={sel.upper()}  dwell={smoother.cfg.min_dwell_s:.2f}s  hold={smoother.cfg.holdoff_s:.2f}s",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if locked:
        cv2.putText(canvas, "LOCKED", (canvas.shape[1] - 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 255), 2, cv2.LINE_AA)

    # dwell bar
    bar_w = canvas.shape[1] - 16
    cv2.rectangle(canvas, (8, h + 44), (8 + bar_w, h + 64), (50, 50, 50), 1)
    fill = int(bar_w * smoother.dwell_progress)
    if fill:
        cv2.rectangle(canvas, (8, h + 44), (8 + fill, h + 64), (40, 190, 255), -1)
    if smoother.candidate:
        cv2.putText(
            canvas,
            f"committing {smoother.candidate}",
            (12, h + 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas
