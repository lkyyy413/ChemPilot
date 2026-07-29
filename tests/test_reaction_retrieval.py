import numpy as np
import pytest

from chempilot.reactions.retrieval import (
    ReactionEmbeddingRetriever,
)


def test_exact_nearest_neighbor():
    index = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = (
        ReactionEmbeddingRetriever(
            index
        )
    )

    indices, similarities = (
        retriever.search(
            [1.0, 0.0],
            top_k=2,
        )
    )

    assert indices.tolist() == [
        [0, 1]
    ]

    assert similarities[
        0,
        0,
    ] == pytest.approx(1.0)


def test_self_exclusion():
    index = np.eye(
        3,
        dtype=np.float32,
    )

    retriever = (
        ReactionEmbeddingRetriever(
            index
        )
    )

    indices, _ = retriever.search(
        index,
        top_k=1,
        excluded_index_by_query=[
            0,
            1,
            2,
        ],
    )

    assert all(
        indices[row, 0] != row
        for row in range(3)
    )


def test_batch_shape():
    index = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = (
        ReactionEmbeddingRetriever(
            index
        )
    )

    indices, similarities = (
        retriever.search(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            top_k=2,
        )
    )

    assert indices.shape == (
        2,
        2,
    )

    assert similarities.shape == (
        2,
        2,
    )


def test_dimension_mismatch():
    retriever = (
        ReactionEmbeddingRetriever(
            np.eye(
                3,
                dtype=np.float32,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        retriever.search(
            [1.0, 0.0]
        )


def test_zero_index_vector_rejected():
    with pytest.raises(
        ValueError,
        match="zero vectors",
    ):
        ReactionEmbeddingRetriever(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        )


def test_zero_query_rejected():
    retriever = (
        ReactionEmbeddingRetriever(
            np.eye(
                2,
                dtype=np.float32,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="zero vectors",
    ):
        retriever.search(
            [0.0, 0.0]
        )


def test_nonfinite_values_rejected():
    with pytest.raises(
        ValueError,
        match="nonfinite",
    ):
        ReactionEmbeddingRetriever(
            [
                [1.0, np.nan],
            ]
        )