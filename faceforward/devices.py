from __future__ import annotations

import sys

import cv2

from .config import Settings, load
from .platform_check import require_windows_host


def opencv_backend(name: str) -> int:
    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    }.get(name.lower(), cv2.CAP_ANY)


def open_camera(index: int, backend: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open a camera with MJPG + size + fps requested *at open time*.

    With DirectShow, setting FOURCC via cap.set() after opening can be silently ignored
    (seen on the Logitech C922/C930e), leaving the camera on uncompressed YUY2 (≈10 fps at 720p, 5 fps at 1080p).
    Passing the properties as open params makes the MJPG request stick (30 fps).
    """
    params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_FRAME_WIDTH, width,
        cv2.CAP_PROP_FRAME_HEIGHT, height,
        cv2.CAP_PROP_FPS, fps,
    ]
    cap = cv2.VideoCapture(index, opencv_backend(backend), params)
    if not cap.isOpened():
        # Some backends reject open params; fall back to setting them after open.
        cap.release()
        cap = cv2.VideoCapture(index, opencv_backend(backend))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def fourcc_str(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def list_dshow_names() -> list[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception as exc:  # pragma: no cover
        print(f"DirectShow name listing unavailable ({exc}). Falling back to indices.", file=sys.stderr)
        return []


def resolve_index(name_contains: str, fallback: int, names: list[str], skip: int | None = None) -> int:
    """First device whose name contains `name_contains` (case-insensitive), ignoring index `skip`."""
    needle = name_contains.lower()
    for i, n in enumerate(names):
        if i != skip and needle in n.lower():
            return i
    return fallback


def resolve_pair(settings: Settings) -> tuple[int, int, list[str]]:
    names = list_dshow_names()
    left = resolve_index(settings.left.name_contains, settings.left.fallback_index, names)
    # Skip the left camera's index so two identical cameras (same name) resolve to
    # the first and second match instead of both landing on the first.
    right = resolve_index(settings.right.name_contains, settings.right.fallback_index, names, skip=left)
    if left == right and names:
        print("WARNING: left and right resolved to the same index. Set fallback_index in config.yaml.", file=sys.stderr)
    return left, right, names


def main() -> None:
    require_windows_host()
    settings = load()
    left, right, names = resolve_pair(settings)
    print("DirectShow cameras:")
    if not names:
        print("  (none listed — install pygrabber, or probe indices 0..n)")
    for i, n in enumerate(names):
        mark = "  <- LEFT" if i == left else "  <- RIGHT" if i == right else ""
        print(f"  [{i}] {n}{mark}")
    print(f"\nResolved  left={left}  right={right}")
    print("Open a preview from each (q to skip):")
    for label, idx in (("LEFT", left), ("RIGHT", right)):
        cap = open_camera(idx, settings.backend, settings.width, settings.height, settings.fps)
        ok, frame = cap.read()
        fmt = fourcc_str(cap)
        cap.release()
        if not ok or frame is None:
            print(f"  {label} index {idx}: FAILED to grab a frame")
            continue
        print(f"  {label} index {idx}: {frame.shape[1]}x{frame.shape[0]} {fmt}")
        cv2.imshow(f"{label} (any key)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
