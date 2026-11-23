"""Multinomial logistic (softmax) regression with L2, pure NumPy.

Ref: Bishop, Pattern Recognition and Machine Learning, 2006, sec. 4.3.4. (README: References)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LABELS = ("left", "right", "away")
LABEL_TO_I = {n: i for i, n in enumerate(LABELS)}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """Multinomial logistic regression. NumPy only — no sklearn."""

    def __init__(self, n_features: int, n_classes: int = 3, l2: float = 1e-3):
        rng = np.random.default_rng(0)
        self.W = rng.normal(0, 0.01, size=(n_features, n_classes))
        self.b = np.zeros((n_classes,), dtype=np.float64)
        self.l2 = l2
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.feature_version: int | None = None  # features.FEATURE_VERSION used in training

    def _norm(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-6
        assert self.mean is not None and self.std is not None
        return (X - self.mean) / self.std

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.3,
        epochs: int = 500,
        sample_weight: np.ndarray | None = None,
    ) -> list[float]:
        """Full-batch gradient descent on weighted cross-entropy + L2.

        sample_weight is normalized to sum to 1 (None = every row equal), so l2 and lr
        mean the same thing regardless of dataset size or weighting scheme.
        Works in X's dtype (float32 input keeps the big matmuls in float32).
        """
        Xn = self._norm(X, fit=True)
        dt = Xn.dtype
        n = Xn.shape[0]
        W = self.W.astype(dt)
        b = self.b.astype(dt)
        Y = np.eye(W.shape[1], dtype=dt)[y]
        w = np.full(n, 1.0 / n, dtype=dt) if sample_weight is None else (sample_weight / sample_weight.sum()).astype(dt)
        losses: list[float] = []
        for _ in range(epochs):
            P = softmax(Xn @ W + b)
            loss = -float(np.sum(w * np.sum(Y * np.log(P + 1e-9), axis=1)))
            loss += 0.5 * self.l2 * float(np.sum(W * W))
            losses.append(loss)
            G = (P - Y) * w[:, None]
            W -= lr * (Xn.T @ G + self.l2 * W)
            b -= lr * G.sum(axis=0)
        self.W = W.astype(np.float64)
        self.b = b.astype(np.float64)
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = self._norm(X, fit=False)
        return softmax(Xn @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            W=self.W,
            b=self.b,
            mean=self.mean,
            std=self.std,
            l2=self.l2,
            feature_version=-1 if self.feature_version is None else self.feature_version,
        )

    @classmethod
    def load(cls, path: Path) -> SoftmaxRegression:
        data = np.load(path, allow_pickle=False)
        m = cls(n_features=data["W"].shape[0], n_classes=data["W"].shape[1], l2=float(data["l2"]))
        m.W = data["W"]
        m.b = data["b"]
        m.mean = data["mean"]
        m.std = data["std"]
        # Weights saved before versioning existed were trained on v1 features.
        m.feature_version = int(data["feature_version"]) if "feature_version" in data.files else 1
        return m
