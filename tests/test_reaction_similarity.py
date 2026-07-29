from pathlib import Path

import pytest

from chempilot.reactions.similarity import (
    SimilarReactionSearch,
)


INDEX = Path(
    "data/processed/reactions/"
    "retrieval/day5/"
    "transformation/cls_index.npz"
)


@pytest.fixture(scope="module")
def searcher():
    if not INDEX.exists():
        pytest.skip(
            "Final retrieval index "
            "not prepared."
        )

    return SimilarReactionSearch(
        protocol="transformation",
        device="cpu",
    )


def test_canonical_reaction():
    (
        reactants,
        products,
        reaction,
    ) = (
        SimilarReactionSearch
        .canonical_reaction(
            [
                "OCC",
                "CCN",
            ],
            [
                "CC=O",
            ],
        )
    )

    assert reactants == [
        "CCN",
        "CCO",
    ]

    assert products == [
        "CC=O",
    ]

    assert reaction == (
        "CCN.CCO>>CC=O"
    )


def test_similarity_search(
    searcher,
):
    result = searcher.search(
        reactant_smiles=[
            "Brc1ccc2ncccc2c1",
            "O=S([O-])C1CC1.[Na+]",
        ],
        product_smiles=[
            (
                "c1cnc2ccc(C3CC3)"
                "cc2c1"
            ),
        ],
        top_k=5,
    )

    assert len(
        result["neighbors"]
    ) == 5

    similarities = [
        neighbor["similarity"]
        for neighbor in (
            result["neighbors"]
        )
    ]

    assert all(
        similarities[index]
        >= similarities[index + 1]
        for index in range(
            len(similarities) - 1
        )
    )

    assert all(
        -1.0 <= value <= 1.0
        for value in similarities
    )

    for neighbor in (
        result["neighbors"]
    ):
        assert (
            "condition_evidence"
            in neighbor
        )


def test_invalid_smiles(
    searcher,
):
    with pytest.raises(
        ValueError,
        match="Invalid SMILES",
    ):
        searcher.search(
            reactant_smiles=[
                "not_a_smiles",
            ],
            product_smiles=[
                "CCO",
            ],
        )


def test_invalid_protocol():
    with pytest.raises(
        ValueError,
        match="protocol",
    ):
        SimilarReactionSearch(
            protocol="invalid"
        )