"""Reusable interfaces for ChemPilot inference components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

import numpy as np


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseFeaturizer(
    ABC,
    Generic[InputT],
):
    """Convert validated inputs into a two-dimensional feature matrix."""

    @property
    @abstractmethod
    def feature_dimension(self) -> int:
        """Return the expected number of output features."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return feature names in their exact model-input order."""

    @abstractmethod
    def transform(
        self,
        inputs: Sequence[InputT],
    ) -> np.ndarray:
        """Transform one batch of inputs."""

    def validate_feature_matrix(
        self,
        features: np.ndarray,
        expected_rows: int,
    ) -> np.ndarray:
        """Validate a generated feature matrix before model inference."""
        matrix = np.asarray(features)

        if matrix.ndim != 2:
            raise ValueError(
                "Feature matrix must be two-dimensional; "
                f"received shape {matrix.shape}."
            )

        expected_shape = (
            expected_rows,
            self.feature_dimension,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "Feature matrix shape mismatch: "
                f"expected {expected_shape}, "
                f"received {matrix.shape}."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Feature matrix contains non-finite values."
            )

        return matrix


class BasePredictor(
    ABC,
    Generic[InputT, OutputT],
):
    """Common interface implemented by production predictors."""

    @abstractmethod
    def predict(
        self,
        inputs: InputT,
    ) -> OutputT:
        """Generate one structured prediction."""