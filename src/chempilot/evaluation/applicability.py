"""Applicability-domain utilities."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _as_binary_float32(
    matrix,
) -> sparse.csr_matrix:
    """Return a binary CSR matrix using float32."""

    result = sparse.csr_matrix(
        matrix,
        dtype=np.float32,
    )

    result.sum_duplicates()

    if result.nnz:
        result.data[:] = 1.0

    return result


def nearest_binary_tanimoto(
    query,
    reference,
    *,
    query_chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest reference row for each query."""

    if query_chunk_size < 1:
        raise ValueError(
            "query_chunk_size must be positive."
        )

    query = _as_binary_float32(query)
    reference = _as_binary_float32(reference)

    if query.shape[1] != reference.shape[1]:
        raise ValueError(
            "Query and reference dimensions differ."
        )

    if reference.shape[0] == 0:
        raise ValueError(
            "Reference matrix is empty."
        )

    query_sizes = np.asarray(
        query.sum(axis=1)
    ).ravel()

    reference_sizes = np.asarray(
        reference.sum(axis=1)
    ).ravel()

    nearest_similarities = np.empty(
        query.shape[0],
        dtype=np.float32,
    )

    nearest_indices = np.empty(
        query.shape[0],
        dtype=np.int64,
    )

    for start in range(
        0,
        query.shape[0],
        query_chunk_size,
    ):
        stop = min(
            start + query_chunk_size,
            query.shape[0],
        )

        intersections = (
            query[start:stop]
            @ reference.T
        ).toarray()

        unions = (
            query_sizes[start:stop, None]
            + reference_sizes[None, :]
            - intersections
        )

        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(
                intersections,
                dtype=np.float32,
            ),
            where=unions > 0,
        )

        local_indices = np.argmax(
            similarities,
            axis=1,
        )

        row_indices = np.arange(
            stop - start
        )

        nearest_indices[start:stop] = (
            local_indices
        )

        nearest_similarities[start:stop] = (
            similarities[
                row_indices,
                local_indices,
            ]
        )

    return (
        nearest_similarities,
        nearest_indices,
    )


def leave_one_out_nearest_tanimoto(
    matrix,
) -> np.ndarray:
    """Return each row's nearest non-self similarity."""

    matrix = _as_binary_float32(matrix)

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two rows are required."
        )

    row_sizes = np.asarray(
        matrix.sum(axis=1)
    ).ravel()

    intersections = (
        matrix @ matrix.T
    ).toarray()

    unions = (
        row_sizes[:, None]
        + row_sizes[None, :]
        - intersections
    )

    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(
            intersections,
            dtype=np.float32,
        ),
        where=unions > 0,
    )

    np.fill_diagonal(
        similarities,
        -np.inf,
    )

    return np.max(
        similarities,
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )