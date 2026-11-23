from __future__ import annotations

import sys
import time

import cv2

from .capture import DualCapture
from .config import load
from .features import FEATURE_DIM, FEATURE_VERSION, learned_vector
from .hysteresis import HysteresisConfig, SwitchSmoother
from .hud import draw_hud
from .model import SoftmaxRegression
from .platform_check import require_windows_host
from .virtual_out import VirtualCam

HUD_HZ = 15  # preview redraw rate; the virtual cam output is paced by camera frames instead


AWAY_THRESHOLD = 0.55  # p_away above this (and the largest) -> treat the frame as "looking away"


def _scores(settings, left, right, model: SoftmaxRegression) -> tuple[float, float]:
    """(p_left, p_right) for one inference-size frame pair; (0, 0) means "away"."""
    x = learned_vector(left, right, settings.left.roi, settings.right.roi)
    p = model.predict_proba(x.reshape(1, -1))[0]
    if p.argmax() == 2 and p[2] > AWAY_THRESHOLD:
        return 0.0, 0.0
    return float(p[0]), float(p[1])


def _load_model(settings) -> SoftmaxRegression:
    """Load weights.npz or exit before any camera is opened."""
    path = settings.weights
    if not path.exists():
        sys.exit(f"No trained model at {path}.\n"
                 "Collect data (py -3 -m faceforward.collect) and train (py -3 -m faceforward.train) first.")
    try:
        model = SoftmaxRegression.load(path)
    except Exception as exc:  # truncated / wrong format
        sys.exit(f"Could not read {path} ({exc}). Re-run py -3 -m faceforward.train.")
    if model.feature_version != FEATURE_VERSION:
        # Same file format, different feature meaning: running it would silently give junk scores.
        sys.exit(f"{path} was trained on feature v{model.feature_version}, this code computes "
                 f"v{FEATURE_VERSION}. Re-run py -3 -m faceforward.train.")
    if model.W.shape != (FEATURE_DIM, 3) or model.mean is None or model.mean.shape != (FEATURE_DIM,):
        sys.exit(f"{path} expects {model.W.shape[0]} features, this code computes {FEATURE_DIM} "
                 "(HOG settings changed?). Re-run py -3 -m faceforward.train.")
    return model


def main() -> None:
    require_windows_host()
    settings = load()
    model = _load_model(settings)

    hy = HysteresisConfig(
        min_dwell_s=settings.min_dwell_s,
        holdoff_s=settings.holdoff_s,
        tau=settings.tau,
        ema_alpha=settings.ema_alpha,
    )
    smoother = SwitchSmoother(hy)
    caps = DualCapture(settings)
    caps.start()
    vcam = VirtualCam(settings.out_width, settings.out_height, settings.out_fps)

    locked = False
    last_infer = 0.0
    s_left = s_right = 0.5
    # Output pacing: only send when the selected camera has produced a new frame.
    last_sent: tuple[str, float] | None = None
    last_hud = 0.0
    hud = None
    # FPS meters (camera ingest vs virtual-cam output), refreshed once per second.
    seen = {"left": 0.0, "right": 0.0}
    counts = {"left": 0, "right": 0, "out": 0}
    rates = {"left": 0.0, "right": 0.0, "out": 0.0}
    meter_t0 = time.perf_counter()
    print("Keys:  [ ] force side   l lock   -/+ dwell   9/0 hold-off   q quit")
    try:
        while True:
            left, right = caps.pair()
            now = time.perf_counter()
            _, stamp_l = caps.left.latest()
            _, stamp_r = caps.right.latest()
            for side, stamp in (("left", stamp_l), ("right", stamp_r)):
                if stamp != seen[side]:
                    seen[side] = stamp
                    counts[side] += 1
            if not locked and (now - last_infer) >= 1.0 / settings.infer_hz:
                inf_l = cv2.resize(left, (settings.infer_width, settings.infer_height))
                inf_r = cv2.resize(right, (settings.infer_width, settings.infer_height))
                sl, sr = _scores(settings, inf_l, inf_r, model)
                s_left, s_right = sl, sr
                if sl + sr > 1e-6:
                    smoother.update(sl, sr, now)
                last_infer = now

            sel = smoother.selected
            sel_stamp = stamp_l if sel == "left" else stamp_r
            if last_sent != (sel, sel_stamp):
                vcam.send(left if sel == "left" else right)
                last_sent = (sel, sel_stamp)
                counts["out"] += 1

            if now - meter_t0 >= 1.0:
                span = now - meter_t0
                rates = {k: v / span for k, v in counts.items()}
                counts = {k: 0 for k in counts}
                meter_t0 = now

            if hud is None or now - last_hud >= 1.0 / HUD_HZ:
                hud = draw_hud(left, right, smoother, s_left, s_right, locked,
                               rois=(settings.left.roi, settings.right.roi))
                cv2.putText(
                    hud,
                    f"cam L {rates['left']:.0f}  R {rates['right']:.0f}  out {rates['out']:.0f} fps",
                    (hud.shape[1] - 230, hud.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("FaceForward", hud)
                last_hud = now
            # waitKey doubles as the loop's idle sleep and keeps the window/keys responsive.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("["):
                smoother.selected = "left"
                smoother.candidate = None
            elif key == ord("]"):
                smoother.selected = "right"
                smoother.candidate = None
            elif key == ord("l"):
                locked = not locked
            elif key in (ord("-"), ord("_")):
                hy.min_dwell_s = max(0.15, hy.min_dwell_s - 0.1)
            elif key in (ord("+"), ord("=")):
                hy.min_dwell_s = min(3.0, hy.min_dwell_s + 0.1)
            elif key == ord("9"):
                hy.holdoff_s = max(0.0, hy.holdoff_s - 0.05)
            elif key == ord("0"):
                hy.holdoff_s = min(1.5, hy.holdoff_s + 0.05)
    finally:
        vcam.close()
        caps.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
