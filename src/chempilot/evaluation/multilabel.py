from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
)


def ensure_dense_binary(
    targets,
) -> np.ndarray:
    if sparse.issparse(targets):
        targets = targets.toarray()

    targets = np.asarray(
        targets,
        dtype=np.uint8,
    )

    if targets.ndim != 2:
        raise ValueError(
            "Targets must be a 2D matrix."
        )

    return targets


def top_k_metrics(
    y_true,
    y_score,
    k: int,
) -> dict:
    y_true = ensure_dense_binary(
        y_true
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float64,
    )

    if y_true.shape != y_score.shape:
        raise ValueError(
            "Target and score shapes differ."
        )

    if not 1 <= k <= y_true.shape[1]:
        raise ValueError(
            f"Invalid k={k}."
        )

    top_indices = np.argpartition(
        -y_score,
        kth=k - 1,
        axis=1,
    )[:, :k]

    hits = np.take_along_axis(
        y_true,
        top_indices,
        axis=1,
    ).sum(axis=1)

    true_counts = y_true.sum(
        axis=1
    )

    valid_rows = true_counts > 0

    if not valid_rows.any():
        return {
            "k": int(k),
            "samples": 0,
            "hit_rate": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    hits = hits[valid_rows]
    true_counts = true_counts[
        valid_rows
    ]

    return {
        "k": int(k),
        "samples": int(
            valid_rows.sum()
        ),
        "hit_rate": float(
            np.mean(hits > 0)
        ),
        "precision": float(
            np.mean(hits / k)
        ),
        "recall": float(
            np.mean(
                hits / true_counts
            )
        ),
    }


def reciprocal_rank(
    y_true,
    y_score,
) -> float:
    y_true = ensure_dense_binary(
        y_true
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float64,
    )

    order = np.argsort(
        -y_score,
        axis=1,
    )

    ranked_targets = np.take_along_axis(
        y_true,
        order,
        axis=1,
    )

    true_counts = y_true.sum(
        axis=1
    )

    valid_rows = true_counts > 0

    reciprocal_ranks = []

    for row in ranked_targets[
        valid_rows
    ]:
        first_positive = int(
            np.flatnonzero(row)[0]
        )

        reciprocal_ranks.append(
            1.0 / (
                first_positive + 1
            )
        )

    return (
        float(
            np.mean(
                reciprocal_ranks
            )
        )
        if reciprocal_ranks
        else float("nan")
    )


def multilabel_metrics(
    y_true,
    y_score,
    top_k_values=(1, 3, 5),
) -> dict:
    y_true = ensure_dense_binary(
        y_true
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float64,
    )

    valid_rows = (
        y_true.sum(axis=1) > 0
    )

    y_true = y_true[
        valid_rows
    ]

    y_score = y_score[
        valid_rows
    ]

    if len(y_true) == 0:
        raise ValueError(
            "No samples contain known "
            "positive labels."
        )

    class_has_positive = (
        y_true.sum(axis=0) > 0
    )

    metrics = {
        "samples": int(
            len(y_true)
        ),
        "classes": int(
            y_true.shape[1]
        ),
        "classes_with_positive_examples": int(
            class_has_positive.sum()
        ),
        "micro_average_precision": float(
            average_precision_score(
                y_true,
                y_score,
                average="micro",
            )
        ),
        "macro_average_precision_observed": float(
            average_precision_score(
                y_true[
                    :,
                    class_has_positive,
                ],
                y_score[
                    :,
                    class_has_positive,
                ],
                average="macro",
            )
        ),
        "mean_reciprocal_rank": (
            reciprocal_rank(
                y_true,
                y_score,
            )
        ),
        "top_k": {},
    }

    for k in top_k_values:
        if k <= y_true.shape[1]:
            metrics["top_k"][
                str(k)
            ] = top_k_metrics(
                y_true,
                y_score,
                k,
            )

    return metrics