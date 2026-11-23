from __future__ import annotations

import cv2
import numpy as np


class VirtualCam:
    """OBS Virtual Camera via pyvirtualcam. Video only — Zoom keeps your real mic."""

    def __init__(self, width: int, height: int, fps: int):
        import pyvirtualcam

        self.width = width
        self.height = height
        # BGR straight from OpenCV: pyvirtualcam converts natively, so no extra full-frame cvtColor pass.
        self._cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera started: {self._cam.device}")

    def send(self, bgr: np.ndarray) -> None:
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._cam.send(bgr)

    def close(self) -> None:
        self._cam.close()
