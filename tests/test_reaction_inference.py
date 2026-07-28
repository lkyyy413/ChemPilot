import numpy as np
import pandas as pd
import pytest

from chempilot.reactions.inference import (
    ReactionConditionPredictor,
)


def normalize(value):
    return (
        ReactionConditionPredictor
        ._normalize_smiles_input(
            value,
            "smiles",
        )
    )


def test_string_smiles_is_accepted():
    assert normalize("CCO") == ["CCO"]


def test_sequence_smiles_is_accepted():
    assert normalize(
        ["CCO", "CCN"]
    ) == [
        "CCO",
        "CCN",
    ]

    assert normalize(
        ("CCO", "CCN")
    ) == [
        "CCO",
        "CCN",
    ]


def test_numpy_smiles_is_accepted():
    value = np.array(
        ["CCO", "CCN"],
        dtype=object,
    )

    assert normalize(value) == [
        "CCO",
        "CCN",
    ]


def test_zero_dimensional_numpy_is_accepted():
    assert normalize(
        np.array("CCO")
    ) == ["CCO"]


def test_pandas_series_is_accepted():
    value = pd.Series(
        ["CCO", "CCN"]
    )

    assert normalize(value) == [
        "CCO",
        "CCN",
    ]


def test_empty_values_are_rejected():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        normalize([])

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        normalize(
            np.array(
                [],
                dtype=object,
            )
        )


def test_invalid_type_is_rejected():
    with pytest.raises(
        TypeError,
        match="received int",
    ):
        normalize(123)