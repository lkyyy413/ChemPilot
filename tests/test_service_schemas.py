import pytest
from pydantic import ValidationError

from chempilot.service.schemas import (
    ErrorDetail,
    ErrorResponse,
    PredictionRequest,
)


def test_molecule_only_request():
    request = PredictionRequest(
        molecule_smiles=" CCO ",
    )

    assert (
        request.molecule_smiles
        == "CCO"
    )

    assert request.reactant_smiles is None
    assert request.product_smiles is None
    assert request.top_k == 5


def test_reaction_only_request():
    request = PredictionRequest(
        reactant_smiles=[
            "CCBr",
            "N",
        ],
        product_smiles=[
            "CCN",
        ],
        reaction_protocol=(
            "transformation"
        ),
        top_k=10,
    )

    assert request.reactant_smiles == [
        "CCBr",
        "N",
    ]

    assert request.product_smiles == [
        "CCN",
    ]

    assert request.top_k == 10


def test_combined_request():
    request = PredictionRequest(
        molecule_smiles="CCO",
        reactant_smiles=[
            "CCBr",
            "N",
        ],
        product_smiles=[
            "CCN",
        ],
    )

    assert (
        request.molecule_smiles
        == "CCO"
    )

    assert (
        request.reactant_smiles
        is not None
    )


def test_empty_request_is_rejected():
    with pytest.raises(
        ValidationError,
        match="Provide molecule_smiles",
    ):
        PredictionRequest()


def test_partial_reaction_is_rejected():
    with pytest.raises(
        ValidationError,
        match=(
            "must be provided together"
        ),
    ):
        PredictionRequest(
            reactant_smiles=[
                "CCBr",
            ],
        )


def test_empty_smiles_is_rejected():
    with pytest.raises(
        ValidationError,
        match="must not be empty",
    ):
        PredictionRequest(
            molecule_smiles="   ",
        )


def test_empty_reaction_item_is_rejected():
    with pytest.raises(
        ValidationError,
        match="must not be empty",
    ):
        PredictionRequest(
            reactant_smiles=[
                "CCBr",
                " ",
            ],
            product_smiles=[
                "CCN",
            ],
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        21,
    ],
)
def test_top_k_range_is_enforced(
    top_k,
):
    with pytest.raises(
        ValidationError,
    ):
        PredictionRequest(
            molecule_smiles="CCO",
            top_k=top_k,
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(
        ValidationError,
        match="Extra inputs",
    ):
        PredictionRequest(
            molecule_smiles="CCO",
            unknown_field=True,
        )


def test_structured_error_response():
    response = ErrorResponse(
        request_id="request-1",
        error=ErrorDetail(
            code="INVALID_SMILES",
            message=(
                "The supplied SMILES "
                "could not be parsed."
            ),
            field="molecule_smiles",
        ),
    )

    serialized = response.model_dump()

    assert serialized["status"] == (
        "error"
    )

    assert serialized[
        "error"
    ]["code"] == "INVALID_SMILES"