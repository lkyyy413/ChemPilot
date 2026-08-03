from pathlib import Path

import joblib
import numpy as np
import pytest

from chempilot.service.errors import (
    InvalidSmilesError,
    ModelArtifactError,
)
from chempilot.service.solubility import (
    DEFAULT_SOLUBILITY_MODEL_PATH,
    DruglikeScopeEvaluator,
    SolubilityFeaturizer,
    SolubilityPredictor,
)


class ConstantSolubilityModel:
    """Small serializable model used by unit tests."""

    n_features_in_ = 2058

    def __init__(
        self,
        prediction: float = -2.5,
    ) -> None:
        self.prediction = prediction

    def predict(
        self,
        features,
    ) -> np.ndarray:
        return np.full(
            len(features),
            self.prediction,
            dtype=np.float32,
        )


class WrongDimensionModel:
    n_features_in_ = 128

    def predict(
        self,
        features,
    ) -> np.ndarray:
        return np.zeros(
            len(features),
            dtype=np.float32,
        )


class NonfinitePredictionModel:
    n_features_in_ = 2058

    def predict(
        self,
        features,
    ) -> np.ndarray:
        return np.full(
            len(features),
            np.nan,
            dtype=np.float32,
        )


def save_model(
    temporary_path: Path,
    model,
) -> Path:
    path = (
        temporary_path
        / "model.joblib"
    )

    joblib.dump(
        model,
        path,
    )

    return path


def test_solubility_feature_contract():
    featurizer = (
        SolubilityFeaturizer()
    )

    features = featurizer.transform(
        [
            "CCO",
            "CC(=O)Oc1ccccc1C(=O)O",
        ]
    )

    assert features.shape == (
        2,
        2058,
    )
    assert features.dtype == np.float32
    assert np.isfinite(
        features
    ).all()

    assert (
        featurizer.feature_names[:10]
        == [
            "molecular_weight",
            "logp",
            "tpsa",
            "hbd",
            "hba",
            "rotatable_bonds",
            "ring_count",
            "fraction_csp3",
            "heavy_atom_count",
            "formal_charge",
        ]
    )

    assert (
        featurizer.feature_names[10]
        == "ecfp_0000"
    )


def test_day1_druglike_scope_rules():
    evaluator = (
        DruglikeScopeEvaluator()
    )

    in_scope = evaluator.evaluate(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    assert in_scope.in_scope
    assert not in_scope.failed_rules

    low_weight = evaluator.evaluate(
        "CCO"
    )

    assert not low_weight.in_scope
    assert (
        "molecular_weight_outside_50_1000"
        in low_weight.failed_rules
    )

    multifragment = evaluator.evaluate(
        "CCO.Cl"
    )

    assert not multifragment.in_scope
    assert (
        "multiple_fragments"
        in multifragment.failed_rules
    )

    no_carbon = evaluator.evaluate(
        "O"
    )

    assert not no_carbon.in_scope
    assert (
        "no_carbon"
        in no_carbon.failed_rules
    )

    uncommon = evaluator.evaluate(
        "C[Sb]"
    )

    assert not uncommon.in_scope
    assert (
        "uncommon_element"
        in uncommon.failed_rules
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "",
        "   ",
        "this-is-not-smiles",
    ],
)
def test_invalid_smiles_is_rejected(
    smiles,
):
    evaluator = (
        DruglikeScopeEvaluator()
    )

    with pytest.raises(
        InvalidSmilesError,
    ):
        evaluator.evaluate(
            smiles
        )


def test_serialized_model_prediction(
    tmp_path,
):
    model_path = save_model(
        tmp_path,
        ConstantSolubilityModel(
            prediction=-2.75,
        ),
    )

    predictor = (
        SolubilityPredictor(
            model_path=model_path,
        )
    )

    result = predictor.predict(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    assert result.predicted_log_s == pytest.approx(
        -2.75
    )
    assert result.unit == (
        "log10(mol/L)"
    )
    assert (
        result.applicability_warning
        is None
    )


def test_out_of_scope_prediction_has_warning(
    tmp_path,
):
    model_path = save_model(
        tmp_path,
        ConstantSolubilityModel(),
    )

    predictor = (
        SolubilityPredictor(
            model_path=model_path,
        )
    )

    result = predictor.predict(
        "CCO"
    )

    assert result.applicability_warning
    assert (
        "molecular_weight_outside_50_1000"
        in result.applicability_warning
    )


def test_missing_model_is_reported(
    tmp_path,
):
    missing_path = (
        tmp_path
        / "missing.joblib"
    )

    with pytest.raises(
        ModelArtifactError,
        match="missing",
    ):
        SolubilityPredictor(
            model_path=missing_path,
        )


def test_corrupt_model_is_reported(
    tmp_path,
):
    corrupt_path = (
        tmp_path
        / "corrupt.joblib"
    )

    corrupt_path.write_bytes(
        b"not a valid joblib artifact"
    )

    with pytest.raises(
        ModelArtifactError,
        match="could not be loaded",
    ):
        SolubilityPredictor(
            model_path=corrupt_path,
        )


def test_model_dimension_mismatch_is_reported(
    tmp_path,
):
    model_path = save_model(
        tmp_path,
        WrongDimensionModel(),
    )

    with pytest.raises(
        ModelArtifactError,
        match="dimension mismatch",
    ):
        SolubilityPredictor(
            model_path=model_path,
        )


def test_nonfinite_model_output_is_reported(
    tmp_path,
):
    model_path = save_model(
        tmp_path,
        NonfinitePredictionModel(),
    )

    predictor = (
        SolubilityPredictor(
            model_path=model_path,
        )
    )

    with pytest.raises(
        ModelArtifactError,
        match="invalid prediction",
    ):
        predictor.predict(
            "CC(=O)Oc1ccccc1C(=O)O"
        )


@pytest.mark.skipif(
    not DEFAULT_SOLUBILITY_MODEL_PATH.is_file(),
    reason=(
        "Generated Day 2 model artifact "
        "is not available."
    ),
)
def test_real_model_matches_historical_prediction():
    predictor = (
        SolubilityPredictor()
    )

    result = predictor.predict(
        "C1CO[Sb]2OCCO[Sb](O1)OCCO2"
    )

    from math import isfinite

    assert isfinite(
        result.predicted_log_s
    )
    assert (
        result.applicability_warning
        is not None
    )

    assert result.applicability_warning
    assert (
        "uncommon_element"
        in result.applicability_warning
    )