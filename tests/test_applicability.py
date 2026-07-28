import numpy as np
import pytest
from scipy import sparse

from chempilot.evaluation.applicability import (
    leave_one_out_nearest_tanimoto,
    nearest_binary_tanimoto,
)


def test_nearest_binary_tanimoto():
    reference = sparse.csr_matrix(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int8,
    )

    query = sparse.csr_matrix(
        [
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int8,
    )

    similarities, indices = (
        nearest_binary_tanimoto(
            query,
            reference,
        )
    )

    assert np.allclose(
        similarities,
        [2.0 / 3.0, 1.0],
    )

    assert indices.tolist() == [0, 2]


def test_leave_one_out_excludes_self():
    matrix = sparse.csr_matrix(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )

    similarities = (
        leave_one_out_nearest_tanimoto(
            matrix
        )
    )

    assert np.allclose(
        similarities,
        [0.5, 0.5, 0.0],
    )


def test_dimension_mismatch_is_rejected():
    query = sparse.csr_matrix(
        [[1, 0]]
    )

    reference = sparse.csr_matrix(
        [[1, 0, 0]]
    )

    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        nearest_binary_tanimoto(
            query,
            reference,
        )


def test_empty_reference_is_rejected():
    query = sparse.csr_matrix(
        [[1, 0]]
    )

    reference = sparse.csr_matrix(
        (0, 2)
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        nearest_binary_tanimoto(
            query,
            reference,
        )


def test_leave_one_out_requires_two_rows():
    matrix = sparse.csr_matrix(
        [[1, 0]]
    )

    with pytest.raises(
        ValueError,
        match="two rows",
    ):
        leave_one_out_nearest_tanimoto(
            matrix
        )