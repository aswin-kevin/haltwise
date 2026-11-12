"""A small, dependency-free (numpy only) L2-regularised logistic regression.

Kept deliberately minimal so the core library needs no scikit-learn. It follows the
scikit-learn conventions (`fit` / `predict_proba`), so swapping in any sklearn model
is a one-line change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _matmul(a: FloatArray, b: FloatArray) -> FloatArray:
    """Matrix product without spurious floating-point warnings.

    numpy 2.2 builds that use Apple Accelerate raise overflow/divide flags inside
    ``matmul`` even when every result is finite. The flags are silenced here, and the
    result is checked explicitly instead, so a genuinely bad result still fails loudly.
    """
    with np.errstate(all="ignore"):
        product = a @ b
    if not np.all(np.isfinite(product)):
        raise FloatingPointError("matrix product produced non-finite values")
    return product


def _sigmoid(z: FloatArray) -> FloatArray:
    # exp(-log(1 + e^-z)) is the overflow-safe form of 1 / (1 + e^-z).
    return np.exp(-np.logaddexp(0.0, -z))


class LogisticRegression:
    """Binary logistic regression fitted with Newton's method.

    Features are standardised internally, so ``l2`` means the same thing regardless
    of feature scale.

    Args:
        l2: Strength of the L2 penalty on the weights (not the intercept).
        max_iter: Maximum Newton steps; a handful is usually enough.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 50) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None
        self.weights_: FloatArray | None = None  # intercept first

    def fit(
        self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> LogisticRegression:
        """Fit to features ``x`` (n x d) and binary labels ``y`` (1 = positive class)."""
        features = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(features)):
            raise ValueError("features must be finite (no NaN or infinity)")
        labels = np.asarray(y, dtype=np.float64)
        weights = (
            np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, float)
        )
        weights = weights * (len(weights) / weights.sum())  # mean weight 1: l2 stays comparable

        self.mean_ = features.mean(axis=0)
        std = features.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        design = self._design(features)

        penalty = np.full(design.shape[1], self.l2)
        penalty[0] = 1e-9  # tiny ridge on the intercept, for numerical stability only
        coef = np.zeros(design.shape[1])
        for _ in range(self.max_iter):
            p = _sigmoid(_matmul(design, coef))
            gradient = _matmul(design.T, weights * (p - labels)) + penalty * coef
            hessian = _matmul(design.T * (weights * p * (1 - p)), design) + np.diag(penalty)
            step = np.linalg.solve(hessian, gradient)
            coef -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.weights_ = coef
        return self

    def predict_proba(self, x: ArrayLike) -> FloatArray:
        """Return an (n x 2) array of [P(negative), P(positive)], like scikit-learn."""
        if self.weights_ is None:
            raise RuntimeError("LogisticRegression is not fitted yet; call fit() first")
        positive = _sigmoid(_matmul(self._design(np.asarray(x, dtype=np.float64)), self.weights_))
        return np.column_stack([1.0 - positive, positive])

    def _design(self, features: FloatArray) -> FloatArray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("LogisticRegression is not fitted yet; call fit() first")
        standardised = (features - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(standardised)), standardised])

    def to_dict(self) -> dict[str, Any]:
        """Serialise the fitted model to plain JSON."""
        if self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("cannot serialise an unfitted LogisticRegression")
        return {
            "l2": self.l2,
            "max_iter": self.max_iter,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "weights": self.weights_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogisticRegression:
        """Rebuild a model saved with `to_dict`."""
        model = cls(l2=data["l2"], max_iter=data["max_iter"])
        model.mean_ = np.asarray(data["mean"], dtype=np.float64)
        model.scale_ = np.asarray(data["scale"], dtype=np.float64)
        model.weights_ = np.asarray(data["weights"], dtype=np.float64)
        return model
