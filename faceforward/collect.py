from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from .capture import DualCapture
from .config import load
from .hud import draw_roi
from .platform_check import require_windows_host

KEYS = {ord("1"): "left", ord("2"): "right", ord("0"): "away"}
CLASSES = ("left", "right", "away")


def _next_id(folder: Path) -> int:
    """Highest existing pair number + 1. (Counting files instead would reuse — and
    overwrite — numbers after any photo is deleted.)"""
    nums = [int(p.name[:-len("_L.jpg")]) for p in folder.glob("*_L.jpg") if p.name[:-len("_L.jpg")].isdigit()]
    return max(nums, default=0) + 1


def _discard_session(root: Path, meta_path: Path, session: str, saved: list[tuple[str, str]]) -> None:
    """Delete this session's photo pairs and its meta.jsonl rows (other sessions untouched)."""
    for label, stem in saved:
        for side in ("L", "R"):
            (root / label / f"{stem}_{side}.jpg").unlink(missing_ok=True)
    if meta_path.exists():
        kept = []
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("session") == session:
                    continue
            except json.JSONDecodeError:
                pass
            kept.append(line + "\n")
        tmp = meta_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        tmp.replace(meta_path)  # atomic swap: meta.jsonl is never left half-written


def main() -> None:
    require_windows_host()
    settings = load()
    spec = settings.collect
    root: Path = settings.dataset_dir
    for label in CLASSES:
        (root / label).mkdir(parents=True, exist_ok=True)
    meta_path = root / "meta.jsonl"

    # One run of collect = one session. train.py weights and validates per session.
    session = time.strftime("%Y%m%d-%H%M%S")
    next_id = {k: _next_id(root / k) for k in CLASSES}
    totals = {k: len(list((root / k).glob("*_L.jpg"))) for k in CLASSES}
    counts = {k: 0 for k in CLASSES}  # this session only
    saved: list[tuple[str, str]] = []  # (label, stem) written this session, for discarding

    caps = DualCapture(settings)
    caps.start()
    print(f"Session {session}. Press 1=facing LEFT  2=RIGHT  0=away/down. Press again or space to stop. q=quit")
    print(f"Goal: {spec.target_per_class} of EACH class this session. Keep moving; vary posture and position.")
    if spec.discard_incomplete:
        print("Quitting before every class reaches the goal DISCARDS this session (collect.discard_incomplete).")
    active: str | None = None
    last_save = 0.0
    try:
        while True:
            left, right = caps.pair()
            pl, pr = cv2.resize(left, (640, 360)), cv2.resize(right, (640, 360))
            draw_roi(pl, settings.left.roi)
            draw_roi(pr, settings.right.roi)
            vis = cv2.hconcat([pl, pr])

            head = f"RECORDING {active.upper()}" if active else "1/2/0 to record"
            cv2.putText(vis, f"{head}   session {session}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (40, 190, 255), 2, cv2.LINE_AA)
            # Per-session progress: green once a class reaches its target.
            x = 12
            for k, short in (("left", "L"), ("right", "R"), ("away", "A")):
                done = counts[k] >= spec.target_per_class
                color = (80, 220, 80) if done else (40, 190, 255)
                text = f"{short} {counts[k]}/{spec.target_per_class}"
                cv2.putText(vis, text, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                x += 170
            cv2.putText(vis, f"all sessions: L={totals['left']} R={totals['right']} A={totals['away']}",
                        (x + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("FaceForward collect", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                active = None
            elif key in KEYS:
                nxt = KEYS[key]
                active = None if active == nxt else nxt

            now = time.perf_counter()
            if active and now - last_save >= 1.0 / spec.save_hz:
                stem = f"{next_id[active]:05d}"
                cv2.imwrite(str(root / active / f"{stem}_L.jpg"), left, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(str(root / active / f"{stem}_R.jpg"), right, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                rec = {
                    "label": active,
                    "id": stem,
                    "t": time.time(),
                    "session": session,
                    "left_index": caps.left_index,
                    "right_index": caps.right_index,
                }
                with meta_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                saved.append((active, stem))
                next_id[active] += 1
                counts[active] += 1
                totals[active] += 1
                last_save = now
    finally:
        caps.stop()
        cv2.destroyAllWindows()
        print(f"session {session}: {counts}")
        missing = [k for k in CLASSES if counts[k] < spec.target_per_class]
        if saved and missing and spec.discard_incomplete:
            _discard_session(root, meta_path, session, saved)
            print(f"Below target for: {', '.join(missing)}. DISCARDED this session: "
                  f"deleted {len(saved)} pairs and their meta.jsonl rows.")
        elif saved and missing:
            print(f"Saved to {root}, but below target for: {', '.join(missing)}. Unbalanced sessions "
                  f"teach the model 'this lighting/outfit = that class'.")
        elif saved:
            print(f"Session complete. Saved {len(saved)} pairs to {root}")


if __name__ == "__main__":
    main()
