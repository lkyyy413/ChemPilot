import pytest

from chempilot.service.errors import (
    InvalidSmilesError,
    ModelArtifactError,
)
from chempilot.service.molecule import (
    MoleculeRiskAnalyzer,
    MoleculeStandardizer,
)
from chempilot.service.prediction import (
    PredictionService,
)
from chempilot.service.schemas import (
    PredictionRequest,
    SolubilityPrediction,
)
from chempilot.service.solubility import (
    DruglikeScopeEvaluator,
)


CONDITION_RESULT = {
    "protocol": "reaction_center",
    "predictions": {
        "solvent": {
            "target": "solvent",
            "top_k": [
                {
                    "rank": 1,
                    "label": "SMILES:CCO",
                    "score": 0.8,
                },
                {
                    "rank": 2,
                    "label": "SMILES:O",
                    "score": 0.6,
                },
            ],
            "score_interpretation": (
                "Uncalibrated ranking score; "
                "not a calibrated probability."
            ),
            "applicability_domain": {
                "in_domain": True,
                "nearest_similarity": 0.55,
                "threshold": 0.2,
                "threshold_definition": (
                    "Training similarity threshold"
                ),
                "nearest_train_transformation": (
                    "A>>B"
                ),
                "nearest_train_reaction_center": (
                    "center-1"
                ),
                "nearest_train_reaction_type": (
                    "TEST"
                ),
            },
        },
        "catalyst": {
            "target": "catalyst",
            "top_k": [
                {
                    "rank": 1,
                    "label": "SMILES:[Pd]",
                    "score": 0.7,
                },
                {
                    "rank": 2,
                    "label": "SMILES:[Ni]",
                    "score": 0.4,
                },
            ],
            "score_interpretation": (
                "Uncalibrated ranking score; "
                "not a calibrated probability."
            ),
            "applicability_domain": {
                "in_domain": True,
                "nearest_similarity": 0.54,
                "threshold": 0.2,
                "threshold_definition": (
                    "Training similarity threshold"
                ),
                "nearest_train_transformation": (
                    "A>>B"
                ),
                "nearest_train_reaction_center": (
                    "center-1"
                ),
                "nearest_train_reaction_type": (
                    "TEST"
                ),
            },
        },
    },
}


RETRIEVAL_RESULT = {
    "protocol": "reaction_center",
    "embedding": (
        "RXNFP bert_pretrained CLS"
    ),
    "neighbors": [
        {
            "rank": 1,
            "similarity": 0.9,
            "transformation_signature": (
                "SMILES:CCBr>>SMILES:CCO"
            ),
            "canonical_reaction": (
                "CCBr>>CCO"
            ),
            "reaction_type": "TEST",
            "reaction_center_signature": (
                "center-1"
            ),
            "condition_evidence": {
                "condition_pairs": 3,
                "represented_experiments": 4,
                "best_historical_condition": {
                    "solvents": [
                        "SMILES:CCO",
                    ],
                    "catalysts": [
                        "SMILES:[Pd]",
                    ],
                    "reagents": [
                        "SMILES:O",
                    ],
                    "temperature_celsius": (
                        80.0
                    ),
                    "reaction_time_hours": (
                        2.0
                    ),
                    "lc_area_percent": 75.0,
                    "replicate_count": 1,
                    "representative_reaction_id": (
                        "ord-test"
                    ),
                },
            },
        }
    ],
}


class FakeSolubilityPredictor:
    def predict(
        self,
        smiles,
    ):
        return SolubilityPrediction(
            predicted_log_s=-2.0,
            model_name="fake_solubility",
            model_protocol="test",
        )


class FakeConditionPredictor:
    def __init__(
        self,
        result=None,
    ):
        self.result = (
            result
            if result is not None
            else CONDITION_RESULT
        )
        self.calls = []

    def predict(
        self,
        reactant_smiles,
        product_smiles,
        *,
        top_k=5,
    ):
        self.calls.append(
            {
                "reactants": (
                    reactant_smiles
                ),
                "products": (
                    product_smiles
                ),
                "top_k": top_k,
            }
        )

        return self.result


class FakeRetrievalSearch:
    def __init__(
        self,
        result=None,
    ):
        self.result = (
            result
            if result is not None
            else RETRIEVAL_RESULT
        )
        self.calls = []

    def search(
        self,
        *,
        reactant_smiles,
        product_smiles,
        top_k=5,
    ):
        self.calls.append(
            {
                "reactants": (
                    reactant_smiles
                ),
                "products": (
                    product_smiles
                ),
                "top_k": top_k,
            }
        )

        return self.result


class FakeRegistry:
    def __init__(
        self,
        *,
        condition_result=None,
        retrieval_result=None,
    ):
        self.standardizer = (
            MoleculeStandardizer()
        )
        self.risk_analyzer = (
            MoleculeRiskAnalyzer()
        )
        self.scope_evaluator = (
            DruglikeScopeEvaluator()
        )
        self.solubility_predictor = (
            FakeSolubilityPredictor()
        )
        self.condition_predictor = (
            FakeConditionPredictor(
                condition_result
            )
        )
        self.retrieval_search = (
            FakeRetrievalSearch(
                retrieval_result
            )
        )

        self.condition_requests = []
        self.retrieval_requests = []

    def get_standardizer(self):
        return self.standardizer

    def get_molecule_risk_analyzer(
        self,
    ):
        return self.risk_analyzer

    def get_scope_evaluator(self):
        return self.scope_evaluator

    def get_solubility_predictor(
        self,
    ):
        return self.solubility_predictor

    def get_condition_predictor(
        self,
        protocol,
    ):
        self.condition_requests.append(
            protocol
        )
        return self.condition_predictor

    def get_retrieval_search(
        self,
        protocol,
    ):
        self.retrieval_requests.append(
            protocol
        )
        return self.retrieval_search


def test_molecule_only_prediction():
    registry = FakeRegistry()
    service = PredictionService(
        registry
    )

    response = service.predict(
        PredictionRequest(
            molecule_smiles=(
                "CC(=O)Oc1ccccc1C(=O)O"
            )
        ),
        request_id="molecule-test",
    )

    assert (
        response.request_id
        == "molecule-test"
    )
    assert response.status == "success"
    assert response.molecule is not None
    assert response.reaction is None

    assert (
        response.molecule
        .solubility.predicted_log_s
        == -2.0
    )

    assert (
        registry.condition_requests
        == []
    )
    assert (
        registry.retrieval_requests
        == []
    )


def test_reaction_only_prediction():
    registry = FakeRegistry()
    service = PredictionService(
        registry
    )

    response = service.predict(
        PredictionRequest(
            reactant_smiles=[
                "CCBr",
            ],
            product_smiles=[
                "CCO",
            ],
            reaction_protocol=(
                "reaction_center"
            ),
            top_k=2,
        ),
        request_id="reaction-test",
    )

    reaction = response.reaction

    assert response.molecule is None
    assert reaction is not None

    assert (
        reaction.canonical_reactants
        == ["CCBr"]
    )
    assert (
        reaction.canonical_products
        == ["CCO"]
    )

    assert (
        reaction.conditions
        .solvent.top_k[0]
        .label
        == "SMILES:CCO"
    )

    assert (
        reaction.conditions
        .solvent.top_k[0]
        .ranking_score
        == pytest.approx(0.8)
    )

    assert (
        reaction.retrieval.pooling
        == "cls"
    )

    assert (
        reaction.retrieval
        .neighbors[0]
        .condition_evidence
        .analytical_response
        == 75.0
    )

    assert (
        reaction.synthesizability
        .historical_similarity
        == pytest.approx(0.9)
    )

    assert (
        reaction.synthesizability
        .condition_uncertainty
        == pytest.approx(0.75)
    )

    assert (
        reaction.confidence
        .calibrated_probability
        is False
    )


def test_combined_molecule_and_reaction_prediction():
    registry = FakeRegistry()
    service = PredictionService(
        registry
    )

    response = service.predict(
        PredictionRequest(
            molecule_smiles=(
                "CC(=O)Oc1ccccc1C(=O)O"
            ),
            reactant_smiles=[
                "CCBr",
            ],
            product_smiles=[
                "CCO",
            ],
            top_k=1,
        )
    )

    assert response.molecule is not None
    assert response.reaction is not None
    assert response.status == "success"


def test_condition_uncertainty_uses_both_targets():
    service = PredictionService(
        FakeRegistry()
    )

    uncertainty = (
        service
        ._condition_uncertainty(
            CONDITION_RESULT
        )
    )

    # Solvent: 1 - (0.8 - 0.6) = 0.8
    # Catalyst: 1 - (0.7 - 0.4) = 0.7
    # Mean = 0.75
    assert uncertainty == pytest.approx(
        0.75
    )


def test_invalid_molecule_smiles_is_rejected():
    service = PredictionService(
        FakeRegistry()
    )

    with pytest.raises(
        InvalidSmilesError,
    ):
        service.predict(
            PredictionRequest(
                molecule_smiles=(
                    "not-a-smiles"
                )
            )
        )


def test_invalid_reactant_smiles_is_rejected():
    service = PredictionService(
        FakeRegistry()
    )

    with pytest.raises(
        InvalidSmilesError,
    ):
        service.predict(
            PredictionRequest(
                reactant_smiles=[
                    "not-a-smiles",
                ],
                product_smiles=[
                    "CCO",
                ],
            )
        )


def test_empty_condition_ranking_is_rejected():
    malformed = {
        "protocol": "reaction_center",
        "predictions": {
            "solvent": {
                "target": "solvent",
                "top_k": [],
            },
            "catalyst": {
                "target": "catalyst",
                "top_k": [],
            },
        },
    }

    registry = FakeRegistry(
        condition_result=malformed
    )
    service = PredictionService(
        registry
    )

    with pytest.raises(
        ModelArtifactError,
        match="empty ranking",
    ):
        service.predict(
            PredictionRequest(
                reactant_smiles=[
                    "CCBr",
                ],
                product_smiles=[
                    "CCO",
                ],
            )
        )


def test_malformed_retrieval_is_rejected():
    registry = FakeRegistry(
        retrieval_result={
            "embedding": "RXNFP CLS",
            "neighbors": None,
        }
    )
    service = PredictionService(
        registry
    )

    with pytest.raises(
        ModelArtifactError,
        match="malformed neighbors",
    ):
        service.predict(
            PredictionRequest(
                reactant_smiles=[
                    "CCBr",
                ],
                product_smiles=[
                    "CCO",
                ],
            )
        )


def test_historical_response_is_not_reported_as_yield():
    service = PredictionService(
        FakeRegistry()
    )

    response = service.predict(
        PredictionRequest(
            reactant_smiles=[
                "CCBr",
            ],
            product_smiles=[
                "CCO",
            ],
        )
    )

    reaction = response.reaction
    assert reaction is not None

    evidence = (
        reaction.retrieval
        .neighbors[0]
        .condition_evidence
    )

    assert evidence is not None
    assert (
        "not isolated reaction yield"
        in evidence
        .response_interpretation
    )

    assert any(
        "not isolated reaction yield"
        in warning
        for warning in (
            reaction.warnings
        )
    )