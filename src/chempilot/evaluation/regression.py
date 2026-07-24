"""Shared evaluation utilities for molecular regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """Calculate the common regression metrics used in ChemPilot."""

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: {y_true.shape} versus {y_pred.shape}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not np.isfinite(y_true).all():
        raise ValueError("Non-finite values found in y_true.")

    if not np.isfinite(y_pred).all():
        raise ValueError("Non-finite values found in y_pred.")

    rmse = float(
        mean_squared_error(
            y_true,
            y_pred,
        ) ** 0.5
    )

    if (
        np.unique(y_true).size < 2
        or np.unique(y_pred).size < 2
    ):
        spearman = float("nan")
    else:
        spearman = float(
            pd.Series(y_true).corr(
                pd.Series(y_pred),
                method="spearman",
            )
        )

    return {
        "n_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman,
    }


def prediction_table(
    sample_ids: np.ndarray,
    smiles: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    in_druglike_scope: np.ndarray,
) -> pd.DataFrame:
    """Create a sample-level prediction and error table."""

    result = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "smiles": smiles,
            "y_true": y_true,
            "y_pred": y_pred,
            "in_druglike_scope": in_druglike_scope,
        }
    )

    result["residual"] = (
        result["y_pred"] - result["y_true"]
    )

    result["absolute_error"] = (
        result["residual"].abs()
    )

    return result.sort_values(
        "absolute_error",
        ascending=False,
    ).reset_index(drop=True)