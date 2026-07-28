import numpy as np
import pytest

from chempilot.reactions.features import (
    ReactionFingerprintConfig,
    ReactionFingerprintFeaturizer,
)


def test_reaction_fingerprint_shapes():
    featurizer = (
        ReactionFingerprintFeaturizer(
            ReactionFingerprintConfig(
                number_of_bits=128
            )
        )
    )

    features = featurizer.transform_one(
        reactant_smiles=[
            "CCO",
            "O=C=O",
        ],
        product_smiles=[
            "CC(=O)O",
        ],
    )

    assert features["reactant"].shape == (
        128,
    )
    assert features["product"].shape == (
        128,
    )
    assert features["difference"].shape == (
        128,
    )
    assert features["concatenated"].shape == (
        256,
    )
    assert features["combined"].shape == (
        384,
    )


def test_difference_fingerprint_values():
    featurizer = (
        ReactionFingerprintFeaturizer(
            ReactionFingerprintConfig(
                number_of_bits=128
            )
        )
    )

    features = featurizer.transform_one(
        ["CCO"],
        ["CC=O"],
    )

    assert set(
        np.unique(
            features["difference"]
        ).tolist()
    ).issubset(
        {-1, 0, 1}
    )


def test_component_order_is_invariant():
    featurizer = (
        ReactionFingerprintFeaturizer(
            ReactionFingerprintConfig(
                number_of_bits=128
            )
        )
    )

    first = featurizer.side_fingerprint(
        ["CCO", "CCN"]
    )

    second = featurizer.side_fingerprint(
        ["CCN", "CCO"]
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_batch_transform():
    featurizer = (
        ReactionFingerprintFeaturizer(
            ReactionFingerprintConfig(
                number_of_bits=128
            )
        )
    )

    features = featurizer.transform(
        reactant_smiles=[
            ["CCO"],
            ["CCN", "O"],
        ],
        product_smiles=[
            ["CC=O"],
            ["CCNC"],
        ],
    )

    assert features["combined"].shape == (
        2,
        384,
    )

    assert features["combined"].dtype == (
        np.int8
    )


def test_invalid_smiles_raises_error():
    featurizer = (
        ReactionFingerprintFeaturizer()
    )

    with pytest.raises(
        ValueError,
        match="Unable to parse SMILES",
    ):
        featurizer.transform_one(
            ["not-a-smiles"],
            ["CCO"],
        )


def test_batch_length_mismatch_raises():
    featurizer = (
        ReactionFingerprintFeaturizer()
    )

    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        featurizer.transform(
            reactant_smiles=[
                ["CCO"],
                ["CCN"],
            ],
            product_smiles=[
                ["CC=O"],
            ],
        )

from chempilot.reactions.features import (
    ConditionFingerprintFeaturizer,
)

def test_condition_feature_shape():
    featurizer = (
        ConditionFingerprintFeaturizer()
    )

    features = featurizer.transform_one(
        solvent_labels=[
            "SMILES:CC#N",
        ],
        catalyst_labels=[
            "NAME:test catalyst",
        ],
        reagent_labels=[],
        temperature_celsius=25.0,
        reaction_time_hours=18.0,
    )

    assert features["combined"].shape == (
        18439,
    )

    assert features["combined"].dtype == (
        np.float32
    )

    assert np.isfinite(
        features["combined"]
    ).all()


def test_missing_condition_indicators():
    featurizer = (
        ConditionFingerprintFeaturizer()
    )

    features = featurizer.transform_one(
        solvent_labels=[],
        catalyst_labels=[],
        reagent_labels=[],
        temperature_celsius=np.nan,
        reaction_time_hours=None,
    )

    expected_numeric = np.asarray(
        [
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        features["numeric"],
        expected_numeric,
    )


def test_condition_hash_is_deterministic():
    featurizer = (
        ConditionFingerprintFeaturizer()
    )

    first = (
        featurizer.categorical_fingerprint(
            ["NAME:palladium catalyst"]
        )
    )

    second = (
        featurizer.categorical_fingerprint(
            ["NAME:palladium catalyst"]
        )
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

    assert first.sum() == 1


def test_smiles_condition_has_molecular_bits():
    featurizer = (
        ConditionFingerprintFeaturizer()
    )

    fingerprint = (
        featurizer.molecular_fingerprint(
            ["SMILES:CC#N"]
        )
    )

    assert fingerprint.sum() > 0

def test_smiles_label_is_not_hashed():
    featurizer = (
        ConditionFingerprintFeaturizer()
    )

    fingerprint = (
        featurizer.categorical_fingerprint(
            ["SMILES:CC#N"]
        )
    )

    assert fingerprint.sum() == 0

def test_name_condition_has_no_molecular_bits():
    featurizer = (
        ConditionFingerprintFeaturizer()
    )

    fingerprint = (
        featurizer.molecular_fingerprint(
            ["NAME:unstructured catalyst"]
        )
    )

    assert fingerprint.sum() == 0