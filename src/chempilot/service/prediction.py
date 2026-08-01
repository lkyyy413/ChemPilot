"""Unified ChemPilot prediction orchestration."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import numpy as np
from rdkit import Chem

from chempilot.service.errors import (
    ModelArtifactError,
)
from chempilot.service.registry import (
    ModelRegistry,
)
from chempilot.service.schemas import (
    ApplicabilityAssessment,
    ConditionRecommendation,
    ConditionTargetPrediction,
    ConfidenceAssessment,
    HistoricalConditionEvidence,
    MoleculeAnalysis,
    PredictionRequest,
    RankedConditionLabel,
    ReactionAnalysis,
    RetrievalResult,
    SimilarReaction,
    UnifiedPredictionResponse,
)


LOGGER = logging.getLogger(__name__)


class PredictionService:
    """Coordinate molecule and reaction inference components."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.registry = (
            registry
            or ModelRegistry()
        )

    @staticmethod
    def _ranking_uncertainty(
        prediction: dict[str, Any],
    ) -> float:
        """Convert the top-ranking margin into a heuristic uncertainty."""
        top_k = prediction.get(
            "top_k"
        )

        if (
            not isinstance(top_k, list)
            or not top_k
        ):
            return 1.0

        first = float(
            top_k[0]["score"]
        )

        if len(top_k) == 1:
            margin = max(
                0.0,
                min(1.0, first),
            )
        else:
            second = float(
                top_k[1]["score"]
            )
            margin = max(
                0.0,
                min(
                    1.0,
                    first - second,
                ),
            )

        return float(
            1.0 - margin
        )

    def _condition_uncertainty(
        self,
        raw_result: dict[str, Any],
    ) -> float:
        predictions = raw_result.get(
            "predictions"
        )

        if not isinstance(
            predictions,
            dict,
        ):
            raise ModelArtifactError(
                "Condition model returned no predictions mapping.",
                field=(
                    "reaction_condition_model"
                ),
            )

        try:
            uncertainties = [
                self._ranking_uncertainty(
                    predictions["solvent"]
                ),
                self._ranking_uncertainty(
                    predictions["catalyst"]
                ),
            ]
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ModelArtifactError(
                "Condition model returned malformed ranking scores.",
                field=(
                    "reaction_condition_model"
                ),
                context={
                    "error_type": (
                        type(error).__name__
                    ),
                },
            ) from error

        return float(
            np.mean(uncertainties)
        )

    @staticmethod
    def _map_applicability(
        raw: dict[str, Any] | None,
    ) -> ApplicabilityAssessment | None:
        if raw is None:
            return None

        nearest_reference = raw.get(
            "nearest_train_transformation"
        )

        reference_details = [
            raw.get(
                "nearest_train_reaction_center"
            ),
            raw.get(
                "nearest_train_reaction_type"
            ),
        ]

        reference_details = [
            str(value)
            for value in reference_details
            if value is not None
        ]

        if reference_details:
            suffix = " | ".join(
                reference_details
            )

            if nearest_reference:
                nearest_reference = (
                    f"{nearest_reference} | "
                    f"{suffix}"
                )
            else:
                nearest_reference = suffix

        threshold_definition = raw.get(
            "threshold_definition",
            "Training-set similarity threshold.",
        )

        return ApplicabilityAssessment(
            in_domain=bool(
                raw["in_domain"]
            ),
            nearest_similarity=float(
                raw["nearest_similarity"]
            ),
            threshold=float(
                raw["threshold"]
            ),
            nearest_reference=(
                nearest_reference
            ),
            interpretation=(
                f"{threshold_definition}. "
                "Structural coverage does not "
                "guarantee prediction correctness."
            ),
        )

    def _map_condition_target(
        self,
        raw: dict[str, Any],
    ) -> ConditionTargetPrediction:
        raw_top_k = raw.get(
            "top_k"
        )

        if (
            not isinstance(raw_top_k, list)
            or not raw_top_k
        ):
            raise ModelArtifactError(
                "Condition model returned an empty ranking.",
                field=(
                    "reaction_condition_model"
                ),
            )

        score_interpretation = raw.get(
            "score_interpretation",
            (
                "Uncalibrated ranking score; "
                "not a calibrated probability."
            ),
        )

        ranked = [
            RankedConditionLabel(
                rank=int(
                    item["rank"]
                ),
                label=str(
                    item["label"]
                ),
                ranking_score=float(
                    item["score"]
                ),
                score_interpretation=(
                    score_interpretation
                ),
            )
            for item in raw_top_k
        ]

        return ConditionTargetPrediction(
            target=raw["target"],
            top_k=ranked,
            applicability=(
                self._map_applicability(
                    raw.get(
                        "applicability_domain"
                    )
                )
            ),
        )

    def _map_conditions(
        self,
        protocol: str,
        raw_result: dict[str, Any],
    ) -> ConditionRecommendation:
        predictions = raw_result.get(
            "predictions"
        )

        if not isinstance(
            predictions,
            dict,
        ):
            raise ModelArtifactError(
                "Condition model returned malformed output.",
                field=(
                    "reaction_condition_model"
                ),
            )

        try:
            solvent = (
                self._map_condition_target(
                    predictions["solvent"]
                )
            )
            catalyst = (
                self._map_condition_target(
                    predictions["catalyst"]
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ModelArtifactError(
                "Condition model returned malformed target output.",
                field=(
                    "reaction_condition_model"
                ),
                context={
                    "error_type": (
                        type(error).__name__
                    ),
                },
            ) from error

        return ConditionRecommendation(
            protocol=protocol,
            solvent=solvent,
            catalyst=catalyst,
            independent_targets=True,
        )

    @staticmethod
    def _map_condition_evidence(
        raw: dict[str, Any] | None,
    ) -> HistoricalConditionEvidence | None:
        if not isinstance(
            raw,
            dict,
        ):
            return None

        best = raw.get(
            "best_historical_condition"
        )

        if not isinstance(
            best,
            dict,
        ):
            return None

        return HistoricalConditionEvidence(
            solvent_labels=[
                str(value)
                for value in best.get(
                    "solvents",
                    [],
                )
            ],
            catalyst_labels=[
                str(value)
                for value in best.get(
                    "catalysts",
                    [],
                )
            ],
            reagent_labels=[
                str(value)
                for value in best.get(
                    "reagents",
                    [],
                )
            ],
            temperature_celsius=(
                best.get(
                    "temperature_celsius"
                )
            ),
            reaction_time_hours=(
                best.get(
                    "reaction_time_hours"
                )
            ),
            analytical_response=(
                best.get(
                    "lc_area_percent"
                )
            ),
        )

    def _map_retrieval(
        self,
        raw_result: dict[str, Any],
    ) -> RetrievalResult:
        raw_neighbors = raw_result.get(
            "neighbors"
        )

        if not isinstance(
            raw_neighbors,
            list,
        ):
            raise ModelArtifactError(
                "Reaction retrieval returned malformed neighbors.",
                field=(
                    "reaction_retrieval"
                ),
            )

        neighbors = [
            SimilarReaction(
                rank=int(
                    neighbor["rank"]
                ),
                similarity=float(
                    neighbor["similarity"]
                ),
                transformation_signature=str(
                    neighbor[
                        "transformation_signature"
                    ]
                ),
                reaction_center_signature=(
                    neighbor.get(
                        "reaction_center_signature"
                    )
                ),
                reaction_type=(
                    neighbor.get(
                        "reaction_type"
                    )
                ),
                condition_evidence=(
                    self._map_condition_evidence(
                        neighbor.get(
                            "condition_evidence"
                        )
                    )
                ),
            )
            for neighbor in raw_neighbors
        ]

        embedding = str(
            raw_result.get(
                "embedding",
                "RXNFP",
            )
        )

        pooling = (
            "cls"
            if "CLS" in embedding.upper()
            else "masked_mean"
        )

        return RetrievalResult(
            pooling=pooling,
            neighbors=neighbors,
        )

    @staticmethod
    def _nearest_similarity(
        retrieval: RetrievalResult,
    ) -> float | None:
        if not retrieval.neighbors:
            return None

        return float(
            retrieval.neighbors[0]
            .similarity
        )

    @staticmethod
    def _all_targets_in_domain(
        conditions: ConditionRecommendation,
    ) -> bool:
        assessments = [
            conditions.solvent.applicability,
            conditions.catalyst.applicability,
        ]

        available = [
            assessment
            for assessment in assessments
            if assessment is not None
        ]

        return bool(
            available
        ) and all(
            assessment.in_domain
            for assessment in available
        )

    @staticmethod
    def _confidence(
        *,
        conditions: ConditionRecommendation,
        nearest_similarity: float | None,
        condition_uncertainty: float,
    ) -> ConfidenceAssessment:
        in_domain = (
            PredictionService
            ._all_targets_in_domain(
                conditions
            )
        )

        basis = [
            (
                "Both solvent and catalyst "
                f"applicability checks in domain: "
                f"{in_domain}."
            ),
            (
                "Nearest RXNFP historical similarity: "
                f"{nearest_similarity:.4f}."
                if nearest_similarity is not None
                else (
                    "No historical neighbor "
                    "similarity was available."
                )
            ),
            (
                "Condition ranking uncertainty "
                f"heuristic: "
                f"{condition_uncertainty:.4f}."
            ),
        ]

        warnings = [
            (
                "Confidence is a qualitative "
                "assessment, not a calibrated "
                "reaction-success probability."
            )
        ]

        if (
            in_domain
            and nearest_similarity is not None
            and nearest_similarity >= 0.7
            and condition_uncertainty < 0.35
        ):
            level = "high"
        elif (
            in_domain
            and nearest_similarity is not None
            and nearest_similarity >= 0.5
            and condition_uncertainty < 0.6
        ):
            level = "moderate"
        else:
            level = "low"

        if not in_domain:
            warnings.append(
                "At least one condition model is "
                "outside its applicability domain."
            )

        if (
            nearest_similarity is None
            or nearest_similarity < 0.5
        ):
            warnings.append(
                "Historical reaction similarity "
                "is limited."
            )

        if condition_uncertainty >= 0.6:
            warnings.append(
                "The leading condition candidates "
                "have limited ranking separation."
            )

        return ConfidenceAssessment(
            level=level,
            calibrated_probability=False,
            basis=basis,
            warnings=warnings,
        )

    def _molecule_analysis(
        self,
        smiles: str,
    ) -> MoleculeAnalysis:
        standardizer = (
            self.registry
            .get_standardizer()
        )
        analyzer = (
            self.registry
            .get_molecule_risk_analyzer()
        )
        predictor = (
            self.registry
            .get_solubility_predictor()
        )

        canonical, molecule = (
            standardizer.standardize(
                smiles,
                field=(
                    "molecule_smiles"
                ),
            )
        )

        descriptors = (
            analyzer.descriptors(
                molecule
            )
        )
        drug_likeness = (
            analyzer.drug_likeness(
                descriptors
            )
        )
        synthesizability = (
            analyzer.synthesizability(
                molecule
            )
        )
        solubility = (
            predictor.predict(
                canonical
            )
        )

        warnings: list[str] = []

        if (
            solubility
            .applicability_warning
        ):
            warnings.append(
                solubility
                .applicability_warning
            )

        if not drug_likeness.lipinski_pass:
            warnings.append(
                "The molecule violates one or "
                "more Lipinski rules."
            )

        return MoleculeAnalysis(
            input_smiles=smiles,
            canonical_smiles=canonical,
            descriptors=descriptors,
            solubility=solubility,
            drug_likeness=(
                drug_likeness
            ),
            synthesizability=(
                synthesizability
            ),
            warnings=warnings,
        )

    def _reaction_analysis(
        self,
        request: PredictionRequest,
    ) -> ReactionAnalysis:
        assert (
            request.reactant_smiles
            is not None
        )
        assert (
            request.product_smiles
            is not None
        )

        standardizer = (
            self.registry
            .get_standardizer()
        )

        canonical_reactants = []
        for index, smiles in enumerate(
            request.reactant_smiles
        ):
            canonical, _ = (
                standardizer.standardize(
                    smiles,
                    field=(
                        "reactant_smiles"
                        f"[{index}]"
                    ),
                )
            )
            canonical_reactants.append(
                canonical
            )

        canonical_products = []
        for index, smiles in enumerate(
            request.product_smiles
        ):
            canonical, _ = (
                standardizer.standardize(
                    smiles,
                    field=(
                        "product_smiles"
                        f"[{index}]"
                    ),
                )
            )
            canonical_products.append(
                canonical
            )

        condition_predictor = (
            self.registry
            .get_condition_predictor(
                request.reaction_protocol
            )
        )

        retrieval_search = (
            self.registry
            .get_retrieval_search(
                request.reaction_protocol
            )
        )

        raw_conditions = (
            condition_predictor.predict(
                reactant_smiles=(
                    canonical_reactants
                ),
                product_smiles=(
                    canonical_products
                ),
                top_k=request.top_k,
            )
        )

        raw_retrieval = (
            retrieval_search.search(
                reactant_smiles=(
                    canonical_reactants
                ),
                product_smiles=(
                    canonical_products
                ),
                top_k=request.top_k,
            )
        )

        conditions = self._map_conditions(
            request.reaction_protocol,
            raw_conditions,
        )
        retrieval = self._map_retrieval(
            raw_retrieval
        )

        condition_uncertainty = (
            self._condition_uncertainty(
                raw_conditions
            )
        )
        nearest_similarity = (
            self._nearest_similarity(
                retrieval
            )
        )

        product_molecule = (
            Chem.MolFromSmiles(
                ".".join(
                    canonical_products
                )
            )
        )

        if product_molecule is None:
            raise ModelArtifactError(
                "Canonical products could not "
                "be reconstructed for risk analysis.",
                field="product_smiles",
            )

        synthesizability = (
            self.registry
            .get_molecule_risk_analyzer()
            .synthesizability(
                product_molecule,
                historical_similarity=(
                    nearest_similarity
                ),
                condition_uncertainty=(
                    condition_uncertainty
                ),
            )
        )

        confidence = self._confidence(
            conditions=conditions,
            nearest_similarity=(
                nearest_similarity
            ),
            condition_uncertainty=(
                condition_uncertainty
            ),
        )

        warnings = [
            (
                "Solvent and catalyst predictions "
                "are independent rankings, not a "
                "jointly optimized condition."
            ),
            (
                "Historical LC area responses are "
                "not isolated reaction yields."
            ),
            *confidence.warnings,
        ]

        return ReactionAnalysis(
            canonical_reactants=(
                canonical_reactants
            ),
            canonical_products=(
                canonical_products
            ),
            conditions=conditions,
            retrieval=retrieval,
            confidence=confidence,
            synthesizability=(
                synthesizability
            ),
            warnings=list(
                dict.fromkeys(
                    warnings
                )
            ),
        )

    def predict(
        self,
        request: PredictionRequest,
        *,
        request_id: str | None = None,
    ) -> UnifiedPredictionResponse:
        identifier = (
            request_id
            or str(uuid4())
        )

        LOGGER.info(
            "Starting prediction request %s",
            identifier,
        )

        molecule = None
        reaction = None

        if request.molecule_smiles is not None:
            molecule = (
                self._molecule_analysis(
                    request.molecule_smiles
                )
            )

        if (
            request.reactant_smiles
            is not None
        ):
            reaction = (
                self._reaction_analysis(
                    request
                )
            )

        model_versions = {
            "solubility": (
                "Day2 scaffold XGBoost combined"
            ),
            "condition_recommendation": (
                "Day4 Morgan logistic"
            ),
            "reaction_retrieval": (
                "Day5 RXNFP bert_pretrained"
            ),
        }

        response = (
            UnifiedPredictionResponse(
                request_id=identifier,
                status="success",
                molecule=molecule,
                reaction=reaction,
                warnings=[],
                model_versions=(
                    model_versions
                ),
            )
        )

        LOGGER.info(
            "Completed prediction request %s",
            identifier,
        )

        return response