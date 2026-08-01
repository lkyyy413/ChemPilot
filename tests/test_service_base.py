import numpy as np
import pytest

from chempilot.service.base import (
    BaseFeaturizer,
)


class ExampleFeaturizer(
    BaseFeaturizer[str]
):
    @property
    def feature_dimension(self) -> int:
        return 2

    @property
    def feature_names(self) -> list[str]:
        return ["length", "contains_c"]

    def transform(
        self,
        inputs,
    ) -> np.ndarray:
        features = np.asarray(
            [
                [
                    len(value),
                    int("C" in value),
                ]
                for value in inputs
            ],
            dtype=np.float32,
        )

        return self.validate_feature_matrix(
            features,
            expected_rows=len(inputs),
        )


def test_feature_contract():
    featurizer = ExampleFeaturizer()

    features = featurizer.transform(
        ["CCO", "O"],
    )

    assert features.shape == (2, 2)
    assert features.dtype == np.float32
    assert featurizer.feature_names == [
        "length",
        "contains_c",
    ]


def test_wrong_feature_dimension_is_rejected():
    featurizer = ExampleFeaturizer()

    with pytest.raises(
        ValueError,
        match="shape mismatch",
    ):
        featurizer.validate_feature_matrix(
            np.zeros(
                (1, 3),
                dtype=np.float32,
            ),
            expected_rows=1,
        )


def test_nonfinite_features_are_rejected():
    featurizer = ExampleFeaturizer()

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        featurizer.validate_feature_matrix(
            np.asarray(
                [[1.0, np.nan]],
                dtype=np.float32,
            ),
            expected_rows=1,
        )