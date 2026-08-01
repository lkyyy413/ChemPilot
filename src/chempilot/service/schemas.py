"""Typed request and response contracts for unified inference."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


RiskLevel = Literal[
    "low",
    "moderate",
    "high",
    "unknown",
]

ConfidenceLevel = Literal[
    "high",
    "moderate",
    "low",
    "unknown",
]

PredictionStatus = Literal[
    "success",
    "partial",
]


class StrictModel(BaseModel):
    """Base schema that rejects undeclared fields."""

    model_config = ConfigDict(
        extra="forbid",
    )


class PredictionRequest(StrictModel):
    """Unified molecule and optional reaction request."""

    molecule_smiles: str | None = None

    reactant_smiles: list[str] | None = None
    product_smiles: list[str] | None = None

    reaction_protocol: Literal[
        "transformation",
        "reaction_center",
    ] = "reaction_center"

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    @field_validator(
        "molecule_smiles"
    )
    @classmethod
    def normalize_optional_smiles(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "molecule_smiles must not "
                "be empty."
            )

        return normalized

    @field_validator(
        "reactant_smiles",
        "product_smiles",
    )
    @classmethod
    def normalize_smiles_list(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        if value is None:
            return None

        if not value:
            raise ValueError(
                "SMILES lists must not "
                "be empty."
            )

        normalized = []

        for item in value:
            cleaned = item.strip()

            if not cleaned:
                raise ValueError(
                    "SMILES list items must "
                    "not be empty."
                )

            normalized.append(cleaned)

        return normalized

    @model_validator(
        mode="after"
    )
    def validate_request_mode(
        self,
    ) -> "PredictionRequest":
        has_molecule = (
            self.molecule_smiles
            is not None
        )

        has_reactants = (
            self.reactant_smiles
            is not None
        )

        has_products = (
            self.product_smiles
            is not None
        )

        if not (
            has_molecule
            or has_reactants
            or has_products
        ):
            raise ValueError(
                "Provide molecule_smiles, "
                "or provide both "
                "reactant_smiles and "
                "product_smiles."
            )

        if (
            has_reactants
            != has_products
        ):
            raise ValueError(
                "reactant_smiles and "
                "product_smiles must be "
                "provided together."
            )

        return self


class MolecularDescriptors(StrictModel):
    molecular_weight: float
    log_p: float
    tpsa: float
    hydrogen_bond_donors: int
    hydrogen_bond_acceptors: int
    rotatable_bonds: int
    ring_count: int
    heavy_atom_count: int
    fraction_csp3: float


class SolubilityPrediction(StrictModel):
    predicted_log_s: float
    unit: Literal[
        "log10(mol/L)"
    ] = "log10(mol/L)"
    model_name: str
    model_protocol: str
    applicability_warning: str | None = None


class DrugLikenessAssessment(StrictModel):
    lipinski_pass: bool
    lipinski_violations: int
    rule_results: dict[str, bool]
    interpretation: str


class SynthesizabilityRisk(StrictModel):
    sa_score: float = Field(
        ge=1.0,
        le=10.0,
    )

    rare_fragment_count: int = Field(
        ge=0,
    )

    molecular_complexity: float = Field(
        ge=0.0,
    )

    historical_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    condition_uncertainty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    risk_level: RiskLevel
    risk_factors: list[str]
    interpretation: str

    disclaimer: str = (
        "This is a lightweight risk assessment, "
        "not a multi-step retrosynthesis plan "
        "or a guarantee of synthetic feasibility."
    )


class MoleculeAnalysis(StrictModel):
    input_smiles: str
    canonical_smiles: str
    descriptors: MolecularDescriptors
    solubility: SolubilityPrediction
    drug_likeness: DrugLikenessAssessment
    synthesizability: SynthesizabilityRisk
    warnings: list[str] = Field(
        default_factory=list,
    )


class RankedConditionLabel(StrictModel):
    rank: int = Field(
        ge=1,
    )
    label: str
    ranking_score: float

    score_interpretation: str = (
        "Uncalibrated ranking score; "
        "not a calibrated probability."
    )


class ApplicabilityAssessment(StrictModel):
    in_domain: bool
    nearest_similarity: float = Field(
        ge=0.0,
        le=1.0,
    )
    threshold: float = Field(
        ge=0.0,
        le=1.0,
    )
    nearest_reference: str | None = None
    interpretation: str


class ConditionTargetPrediction(StrictModel):
    target: Literal[
        "solvent",
        "catalyst",
    ]
    top_k: list[
        RankedConditionLabel
    ]
    applicability: (
        ApplicabilityAssessment
        | None
    ) = None


class ConditionRecommendation(StrictModel):
    protocol: Literal[
        "transformation",
        "reaction_center",
    ]
    solvent: ConditionTargetPrediction
    catalyst: ConditionTargetPrediction

    independent_targets: bool = True

    interpretation: str = (
        "Solvent and catalyst are predicted "
        "independently and do not constitute "
        "a jointly optimized condition."
    )


class HistoricalConditionEvidence(StrictModel):
    solvent_labels: list[str] = Field(
        default_factory=list,
    )
    catalyst_labels: list[str] = Field(
        default_factory=list,
    )
    reagent_labels: list[str] = Field(
        default_factory=list,
    )
    temperature_celsius: float | None = None
    reaction_time_hours: float | None = None
    analytical_response: float | None = None

    response_interpretation: str = (
        "ORD LC area percent at 280 nm; "
        "not isolated reaction yield."
    )


class SimilarReaction(StrictModel):
    rank: int = Field(
        ge=1,
    )
    similarity: float = Field(
        ge=0.0,
        le=1.0,
    )
    transformation_signature: str
    reaction_center_signature: str | None = None
    reaction_type: str | None = None
    condition_evidence: (
        HistoricalConditionEvidence
        | None
    ) = None


class RetrievalResult(StrictModel):
    pooling: str
    neighbors: list[
        SimilarReaction
    ]

    similarity_interpretation: str = (
        "Cosine similarity in RXNFP embedding "
        "space; not a probability of reaction "
        "success."
    )


class ConfidenceAssessment(StrictModel):
    level: ConfidenceLevel
    calibrated_probability: bool = False
    basis: list[str]
    warnings: list[str] = Field(
        default_factory=list,
    )


class ReactionAnalysis(StrictModel):
    canonical_reactants: list[str]
    canonical_products: list[str]
    conditions: ConditionRecommendation
    retrieval: RetrievalResult
    confidence: ConfidenceAssessment
    synthesizability: SynthesizabilityRisk
    warnings: list[str] = Field(
        default_factory=list,
    )


class UnifiedPredictionResponse(StrictModel):
    request_id: str
    status: PredictionStatus

    molecule: MoleculeAnalysis | None = None
    reaction: ReactionAnalysis | None = None

    warnings: list[str] = Field(
        default_factory=list,
    )

    model_versions: dict[
        str,
        str,
    ] = Field(
        default_factory=dict,
    )


class ErrorDetail(StrictModel):
    code: str
    message: str
    field: str | None = None
    context: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class ErrorResponse(StrictModel):
    request_id: str | None = None
    status: Literal["error"] = "error"
    error: ErrorDetail