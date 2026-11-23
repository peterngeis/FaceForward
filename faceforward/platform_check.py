from __future__ import annotations

import os
import sys


def is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        ver = open("/proc/version", encoding="utf-8").read().lower()
    except OSError:
        return False
    return "microsoft" in ver or "wsl" in ver


def require_windows_host() -> None:
    """Exit unless running on native Windows (DirectShow cameras + OBS virtual camera)."""
    if sys.platform == "win32":
        return
    if is_wsl():
        reason = (
            "FaceForward is running inside WSL.\n"
            "USB cameras and Zoom live on Windows, so a Linux virtual camera\n"
            "will never appear in Zoom/Teams/Discord."
        )
    else:
        reason = (
            f"FaceForward only supports Windows (detected: {sys.platform}).\n"
            "Camera matching and capture use DirectShow, which is Windows-only."
        )
    print(
        f"\n{reason}\n\n"
        "Use native Windows Python (PowerShell or Windows Terminal):\n"
        "  py -3 -m pip install -r DEPENDENCIES.txt\n"
        "  py -3 -m faceforward.devices\n"
        "  py -3 -m faceforward.run\n",
        file=sys.stderr,
    )
    sys.exit(2)
