import numpy as np
import pytest
from scipy import sparse

from chempilot.evaluation.multilabel import (
    multilabel_metrics,
    reciprocal_rank,
    top_k_metrics,
)


def test_top_one_metrics():
    y_true = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    y_score = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.7],
        ],
        dtype=np.float64,
    )

    metrics = top_k_metrics(
        y_true,
        y_score,
        k=1,
    )

    assert metrics["hit_rate"] == (
        pytest.approx(1.0)
    )

    assert metrics["precision"] == (
        pytest.approx(1.0)
    )

    assert metrics["recall"] == (
        pytest.approx(0.75)
    )


def test_reciprocal_rank():
    y_true = np.asarray(
        [
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )

    y_score = np.asarray(
        [
            [0.9, 0.2, 0.1],
            [0.9, 0.8, 0.7],
        ],
        dtype=np.float64,
    )

    value = reciprocal_rank(
        y_true,
        y_score,
    )

    assert value == pytest.approx(
        (
            1.0
            + 1.0 / 3.0
        )
        / 2.0
    )


def test_sparse_targets_supported():
    y_true = sparse.csr_matrix(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    y_score = np.asarray(
        [
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )

    metrics = multilabel_metrics(
        y_true,
        y_score,
        top_k_values=(1,),
    )

    assert (
        metrics["micro_average_precision"]
        == pytest.approx(1.0)
    )


def test_invalid_k_raises():
    with pytest.raises(
        ValueError,
        match="Invalid",
    ):
        top_k_metrics(
            np.asarray([[1, 0]]),
            np.asarray([[0.9, 0.1]]),
            k=3,
        )