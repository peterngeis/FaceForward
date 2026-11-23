from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .config import Settings
from .devices import fourcc_str, open_camera, resolve_pair

class CameraThread:
    def __init__(self, index: int, width: int, height: int, fps: int, backend: str):
        self.index = index
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._ok = False
        self._stop = threading.Event()
        self.cap = open_camera(index, backend, width, height, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cap.isOpened():
            fmt = fourcc_str(self.cap)
            print(f"camera {index}: {fmt} {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            if fmt != "MJPG":
                print(f"  WARNING: camera {index} is not MJPG; expect ~10 fps at 720p.")
        self._thread = threading.Thread(target=self._loop, name=f"cam-{index}", daemon=True)

    def start(self) -> None:
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            with self._lock:
                self._ok = bool(ok)
                if ok:
                    self._frame = frame
                    self._stamp = time.perf_counter()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._frame is None:
                return None, self._stamp
            return self._frame, self._stamp

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class DualCapture:
    def __init__(self, settings: Settings):
        left_i, right_i, names = resolve_pair(settings)
        self.names = names
        self.left_index = left_i
        self.right_index = right_i
        self.left = CameraThread(left_i, settings.width, settings.height, settings.fps, settings.backend)
        self.right = CameraThread(right_i, settings.width, settings.height, settings.fps, settings.backend)

    def start(self) -> None:
        self.left.start()
        self.right.start()
        # warmup
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            lf, _ = self.left.latest()
            rf, _ = self.right.latest()
            if lf is not None and rf is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("Cameras opened but produced no frames. Check indices with python -m faceforward.devices")

    def pair(self) -> tuple[np.ndarray, np.ndarray]:
        lf, _ = self.left.latest()
        rf, _ = self.right.latest()
        if lf is None or rf is None:
            raise RuntimeError("Missing frame from one camera")
        return lf, rf

    def stop(self) -> None:
        self.left.stop()
        self.right.stop()
