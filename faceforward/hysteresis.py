"""Glance filter: EMA-smoothed scores + margin / hold-off / dwell hysteresis.

EMA = exponential smoothing (Brown, 1963). The switch rules are project-specific. (README: References)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HysteresisConfig:
    min_dwell_s: float = 1.0
    holdoff_s: float = 0.4
    tau: float = 0.12
    ema_alpha: float = 0.35


class SwitchSmoother:
    """Glance filter: EMA + margin + dwell + post-switch hold-off."""

    def __init__(self, cfg: HysteresisConfig, initial: str = "left"):
        self.cfg = cfg
        self.selected = initial
        self.ema_left = 0.5
        self.ema_right = 0.5
        self.candidate: str | None = None
        self.candidate_since: float | None = None
        self.last_switch_t: float | None = None
        self.dwell_progress = 0.0

    def update(self, s_left: float, s_right: float, t: float) -> str:
        a = self.cfg.ema_alpha
        self.ema_left = (1.0 - a) * self.ema_left + a * float(s_left)
        self.ema_right = (1.0 - a) * self.ema_right + a * float(s_right)

        margin = abs(self.ema_left - self.ema_right)
        raw = "left" if self.ema_left >= self.ema_right else "right"

        if margin < self.cfg.tau:
            self.candidate = None
            self.candidate_since = None
            self.dwell_progress = 0.0
            return self.selected

        if raw == self.selected:
            self.candidate = None
            self.candidate_since = None
            self.dwell_progress = 0.0
            return self.selected

        if self.last_switch_t is not None and (t - self.last_switch_t) < self.cfg.holdoff_s:
            self.dwell_progress = 0.0
            return self.selected

        if self.candidate != raw:
            self.candidate = raw
            self.candidate_since = t
            self.dwell_progress = 0.0
            return self.selected

        elapsed = t - (t if self.candidate_since is None else self.candidate_since)
        dwell = max(self.cfg.min_dwell_s, 1e-6)
        self.dwell_progress = min(1.0, elapsed / dwell)
        if elapsed >= dwell:
            self.selected = raw
            self.last_switch_t = t
            self.candidate = None
            self.candidate_since = None
            self.dwell_progress = 0.0
        return self.selected
