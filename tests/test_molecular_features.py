import numpy as np
import pytest

from chempilot.features.molecular import (
    CombinedFeaturizer,
    ECFPFeaturizer,
    RDKitDescriptorFeaturizer,
)


def test_descriptor_shape_and_values():
    featurizer = RDKitDescriptorFeaturizer()
    features = featurizer.transform(
        ["CCO", "c1ccccc1"]
    )

    assert features.shape == (2, 10)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_ecfp_is_binary():
    featurizer = ECFPFeaturizer(
        radius=2,
        n_bits=2048,
    )
    features = featurizer.transform(
        ["CCO", "CC(=O)O"]
    )

    assert features.shape == (2, 2048)
    assert features.dtype == np.uint8
    assert np.isin(features, [0, 1]).all()
    assert (features.sum(axis=1) > 0).all()


def test_combined_feature_dimension():
    featurizer = CombinedFeaturizer()
    features = featurizer.transform(["CCO"])

    assert features.shape == (1, 2058)
    assert np.isfinite(features).all()


def test_invalid_smiles_raises_error():
    featurizer = RDKitDescriptorFeaturizer()

    with pytest.raises(ValueError):
        featurizer.transform(["not_a_smiles"])