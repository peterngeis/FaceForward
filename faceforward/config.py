from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass
class CamSpec:
    name_contains: str
    fallback_index: int
    fov_deg: float  # informational (C922 78°, C930e 90°); not used by the model
    # Head region (x0, x1, y0, y1) as frame fractions; the learned model only looks inside it.
    roi: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)


@dataclass
class AugmentSpec:
    """Training-only augmentation ranges (see augment.py). copies=0 disables augmentation."""

    copies: int = 3
    gamma: tuple[float, float] = (0.6, 1.6)
    gain: float = 0.3
    contrast: float = 0.25
    white_balance: float = 0.15
    scale: float = 0.12
    shift: float = 0.08
    rotate_deg: float = 5.0

    def tag(self) -> str:
        return (f"aug{self.copies}|g{self.gamma[0]}-{self.gamma[1]}|k{self.gain}|c{self.contrast}"
                f"|wb{self.white_balance}|s{self.scale}|t{self.shift}|r{self.rotate_deg}")


@dataclass
class CollectSpec:
    save_hz: float = 1.0  # pairs saved per second while a label is active
    target_per_class: int = 100  # per-session goal shown in the collect window
    discard_incomplete: bool = True  # quitting before every class hits the goal deletes the session


@dataclass
class TrainSpec:
    l2: float = 0.1
    weighting: str = "session_class"  # session_class | class | none
    min_session_classes: int = 2  # sessions with fewer classes are evaluated but not trained on
    validation: str = "loso"  # loso (leave-one-session-out) | newest | none
    epochs: int = 400
    lr: float = 0.25


@dataclass
class Settings:
    left: CamSpec
    right: CamSpec
    width: int
    height: int
    fps: int
    backend: str
    out_width: int
    out_height: int
    out_fps: int
    infer_width: int
    infer_height: int
    infer_hz: float
    min_dwell_s: float
    holdoff_s: float
    tau: float
    ema_alpha: float
    dataset_dir: Path
    weights: Path
    augment: AugmentSpec
    collect: CollectSpec
    train: TrainSpec

    @property
    def root(self) -> Path:
        return ROOT


def _cam(d: dict[str, Any]) -> CamSpec:
    return CamSpec(
        name_contains=str(d["name_contains"]),
        fallback_index=int(d["fallback_index"]),
        fov_deg=float(d.get("fov_deg", 0.0)),
        roi=_roi(d.get("roi", (0.0, 1.0, 0.0, 1.0))),
    )


def _roi(v: Any) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = (float(t) for t in v)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"roi must be [x0, x1, y0, y1] with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {v}")
    return (x0, x1, y0, y1)


def _augment(d: dict[str, Any] | None) -> AugmentSpec:
    d = d or {}
    base = AugmentSpec()
    gamma = d.get("gamma", base.gamma)
    return AugmentSpec(
        copies=int(d.get("copies", base.copies)),
        gamma=(float(gamma[0]), float(gamma[1])),
        gain=float(d.get("gain", base.gain)),
        contrast=float(d.get("contrast", base.contrast)),
        white_balance=float(d.get("white_balance", base.white_balance)),
        scale=float(d.get("scale", base.scale)),
        shift=float(d.get("shift", base.shift)),
        rotate_deg=float(d.get("rotate_deg", base.rotate_deg)),
    )


def _collect(d: dict[str, Any] | None) -> CollectSpec:
    d = d or {}
    base = CollectSpec()
    return CollectSpec(
        save_hz=float(d.get("save_hz", base.save_hz)),
        target_per_class=int(d.get("target_per_class", base.target_per_class)),
        discard_incomplete=bool(d.get("discard_incomplete", base.discard_incomplete)),
    )


def _train(d: dict[str, Any] | None) -> TrainSpec:
    d = d or {}
    base = TrainSpec()
    spec = TrainSpec(
        l2=float(d.get("l2", base.l2)),
        weighting=str(d.get("weighting", base.weighting)),
        min_session_classes=int(d.get("min_session_classes", base.min_session_classes)),
        validation=str(d.get("validation", base.validation)),
        epochs=int(d.get("epochs", base.epochs)),
        lr=float(d.get("lr", base.lr)),
    )
    if spec.weighting not in ("session_class", "class", "none"):
        raise ValueError(f"train.weighting must be session_class | class | none, got {spec.weighting!r}")
    if spec.validation not in ("loso", "newest", "none"):
        raise ValueError(f"train.validation must be loso | newest | none, got {spec.validation!r}")
    return spec


def load(path: Path | None = None) -> Settings:
    p = path or DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    cap = raw["capture"]
    out = raw["output"]
    inf = raw["infer"]
    hy = raw["hysteresis"]
    return Settings(
        left=_cam(raw["left"]),
        right=_cam(raw["right"]),
        width=int(cap["width"]),
        height=int(cap["height"]),
        fps=int(cap["fps"]),
        backend=str(cap["backend"]),
        out_width=int(out["width"]),
        out_height=int(out["height"]),
        out_fps=int(out["fps"]),
        infer_width=int(inf["width"]),
        infer_height=int(inf["height"]),
        infer_hz=float(inf["hz"]),
        min_dwell_s=float(hy["min_dwell_s"]),
        holdoff_s=float(hy["holdoff_s"]),
        tau=float(hy["tau"]),
        ema_alpha=float(hy["ema_alpha"]),
        dataset_dir=(ROOT / raw["dataset_dir"]).resolve(),
        weights=(ROOT / raw["weights"]).resolve(),
        augment=_augment(raw.get("augment")),
        collect=_collect(raw.get("collect")),
        train=_train(raw.get("train")),
    )
