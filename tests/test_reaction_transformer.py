from pathlib import Path

import numpy as np
import pytest

from chempilot.reactions.transformer import (
    ReactionTransformerConfig,
    ReactionTransformerEncoder,
)


CHECKPOINT = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)


def test_invalid_max_length():
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        ReactionTransformerConfig(
            max_length=513
        )


def test_invalid_batch_size():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        ReactionTransformerConfig(
            batch_size=0
        )


@pytest.fixture(scope="module")
def encoder():
    if not CHECKPOINT.exists():
        pytest.skip(
            "RXNFP checkpoint not prepared"
        )

    return ReactionTransformerEncoder(
        ReactionTransformerConfig(
            checkpoint_directory=(
                CHECKPOINT
            ),
            max_length=256,
            batch_size=2,
        )
    )


def test_cls_and_mean_shapes(encoder):
    reactions = [
        "CCO>>CC=O",
        "CC(=O)O>>CC(=O)N",
        "Brc1ccccc1>>Oc1ccccc1",
    ]

    cls_embeddings = encoder.encode(
        reactions,
        pooling="cls",
    )

    mean_embeddings = encoder.encode(
        reactions,
        pooling="masked_mean",
    )

    assert cls_embeddings.shape == (
        3,
        256,
    )

    assert mean_embeddings.shape == (
        3,
        256,
    )

    assert cls_embeddings.dtype == (
        np.float32
    )

    assert np.isfinite(
        cls_embeddings
    ).all()

    assert np.isfinite(
        mean_embeddings
    ).all()

    assert not np.allclose(
        cls_embeddings,
        mean_embeddings,
    )


def test_encoding_is_deterministic(
    encoder,
):
    reaction = "CCO>>CC=O"

    first = encoder.encode(reaction)
    second = encoder.encode(reaction)

    np.testing.assert_allclose(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_batching_is_consistent(
    encoder,
):
    reactions = [
        "CCO>>CC=O",
        "CC(=O)O>>CC(=O)N",
        "Brc1ccccc1>>Oc1ccccc1",
    ]

    one_batch = encoder.encode(
        reactions,
        batch_size=3,
    )

    several_batches = encoder.encode(
        reactions,
        batch_size=1,
    )

    np.testing.assert_allclose(
        one_batch,
        several_batches,
        rtol=1e-5,
        atol=1e-5,
    )


def test_l2_normalization(encoder):
    embeddings = encoder.encode(
        [
            "CCO>>CC=O",
            "CCN>>CC=N",
        ],
        normalize=True,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    np.testing.assert_allclose(
        norms,
        np.ones(2),
        rtol=1e-5,
        atol=1e-5,
    )


def test_numpy_input(encoder):
    reactions = np.asarray(
        [
            "CCO>>CC=O",
            "CCN>>CC=N",
        ],
        dtype=object,
    )

    embeddings = encoder.encode(
        reactions
    )

    assert embeddings.shape == (
        2,
        256,
    )


def test_invalid_item_type(encoder):
    with pytest.raises(
        TypeError,
        match="item 1 received int",
    ):
        encoder.encode(
            [
                "CCO>>CC=O",
                123,
            ]
        )


def test_empty_string(encoder):
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        encoder.encode("   ")