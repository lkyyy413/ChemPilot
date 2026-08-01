"""Production aqueous-solubility inference components."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from chempilot.features.molecular import (
    CombinedFeaturizer,
)
from chempilot.service.base import (
    BaseFeaturizer,
    BasePredictor,
)
from chempilot.service.errors import (
    InvalidSmilesError,
    ModelArtifactError,
)
from chempilot.service.schemas import (
    SolubilityPrediction,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_SOLUBILITY_MODEL_PATH = Path(
    "artifacts/models/day2/"
    "scaffold_xgboost_combined.joblib"
)

COMMON_DRUGLIKE_ATOMIC_NUMBERS = frozenset(
    {
        1,   # H
        5,   # B
        6,   # C
        7,   # N
        8,   # O
        9,   # F
        14,  # Si
        15,  # P
        16,  # S
        17,  # Cl
        34,  # Se
        35,  # Br
        53,  # I
    }
)


@dataclass(frozen=True)
class DruglikeScopeResult:
    """Day 1 applicability-scope result for one molecule."""

    canonical_smiles: str
    in_scope: bool
    failed_rules: tuple[str, ...]
    fragment_count: int
    molecular_weight: float
    contains_carbon: bool
    has_uncommon_element: bool


@dataclass
class DruglikeScopeEvaluator:
    """Reproduce the Day 1 AqSolDB applicability rules."""

    minimum_molecular_weight: float = 50.0
    maximum_molecular_weight: float = 1000.0

    def evaluate(
        self,
        smiles: str,
    ) -> DruglikeScopeResult:
        if (
            not isinstance(smiles, str)
            or not smiles.strip()
        ):
            raise InvalidSmilesError(
                "SMILES must be a non-empty string.",
                field="smiles",
                context={
                    "received_type": (
                        type(smiles).__name__
                    ),
                },
            )

        molecule = Chem.MolFromSmiles(
            smiles.strip()
        )

        if molecule is None:
            raise InvalidSmilesError(
                "RDKit failed to parse the supplied SMILES.",
                field="smiles",
                context={
                    "smiles": smiles,
                },
            )

        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )

        atomic_numbers = {
            atom.GetAtomicNum()
            for atom in molecule.GetAtoms()
        }

        fragment_count = len(
            Chem.GetMolFrags(molecule)
        )

        molecular_weight = float(
            Descriptors.MolWt(molecule)
        )

        contains_carbon = (
            6 in atomic_numbers
        )

        has_uncommon_element = (
            not atomic_numbers.issubset(
                COMMON_DRUGLIKE_ATOMIC_NUMBERS
            )
        )

        failed_rules: list[str] = []

        if fragment_count != 1:
            failed_rules.append(
                "multiple_fragments"
            )

        if not contains_carbon:
            failed_rules.append(
                "no_carbon"
            )

        if has_uncommon_element:
            failed_rules.append(
                "uncommon_element"
            )

        if not (
            self.minimum_molecular_weight
            <= molecular_weight
            <= self.maximum_molecular_weight
        ):
            failed_rules.append(
                "molecular_weight_outside_50_1000"
            )

        return DruglikeScopeResult(
            canonical_smiles=canonical_smiles,
            in_scope=not failed_rules,
            failed_rules=tuple(
                failed_rules
            ),
            fragment_count=fragment_count,
            molecular_weight=molecular_weight,
            contains_carbon=contains_carbon,
            has_uncommon_element=(
                has_uncommon_element
            ),
        )


class SolubilityFeaturizer(
    BaseFeaturizer[str]
):
    """Build the exact Day 2 descriptor-first combined features."""

    def __init__(self) -> None:
        self._delegate = CombinedFeaturizer(
            radius=2,
            n_bits=2048,
            include_chirality=True,
        )

    @property
    def feature_dimension(self) -> int:
        return 2058

    @property
    def feature_names(self) -> list[str]:
        return self._delegate.feature_names

    def transform(
        self,
        inputs: Sequence[str],
    ) -> np.ndarray:
        if not inputs:
            raise ValueError(
                "At least one SMILES input is required."
            )

        try:
            features = (
                self._delegate.transform(
                    inputs
                )
            )
        except ValueError as error:
            raise InvalidSmilesError(
                str(error),
                field="smiles",
            ) from error

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        return self.validate_feature_matrix(
            features,
            expected_rows=len(inputs),
        )


class SolubilityPredictor(
    BasePredictor[
        str,
        SolubilityPrediction,
    ]
):
    """Predict AqSolDB LogS using the selected Day 2 model."""

    def __init__(
        self,
        model_path: str | Path = (
            DEFAULT_SOLUBILITY_MODEL_PATH
        ),
        *,
        featurizer: (
            SolubilityFeaturizer | None
        ) = None,
        scope_evaluator: (
            DruglikeScopeEvaluator | None
        ) = None,
    ) -> None:
        self.model_path = Path(
            model_path
        )
        self.featurizer = (
            featurizer
            or SolubilityFeaturizer()
        )
        self.scope_evaluator = (
            scope_evaluator
            or DruglikeScopeEvaluator()
        )
        self.model = self._load_model()

    def _load_model(self) -> Any:
        if not self.model_path.is_file():
            raise ModelArtifactError(
                "Solubility model artifact is missing.",
                field="solubility_model",
                context={
                    "path": str(
                        self.model_path
                    ),
                },
            )

        try:
            model = joblib.load(
                self.model_path
            )
        except Exception as error:
            raise ModelArtifactError(
                "Solubility model artifact could not be loaded.",
                field="solubility_model",
                context={
                    "path": str(
                        self.model_path
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                },
            ) from error

        if not callable(
            getattr(model, "predict", None)
        ):
            raise ModelArtifactError(
                "Solubility artifact does not provide predict().",
                field="solubility_model",
                context={
                    "path": str(
                        self.model_path
                    ),
                    "artifact_type": (
                        type(model).__name__
                    ),
                },
            )

        model_dimension = getattr(
            model,
            "n_features_in_",
            None,
        )

        if model_dimension is None:
            raise ModelArtifactError(
                "Solubility model does not record n_features_in_.",
                field="solubility_model",
                context={
                    "path": str(
                        self.model_path
                    ),
                },
            )

        model_dimension = int(
            model_dimension
        )

        if (
            model_dimension
            != self.featurizer.feature_dimension
        ):
            raise ModelArtifactError(
                "Solubility model feature dimension mismatch.",
                field="solubility_model",
                context={
                    "path": str(
                        self.model_path
                    ),
                    "model_dimension": (
                        model_dimension
                    ),
                    "featurizer_dimension": (
                        self.featurizer
                        .feature_dimension
                    ),
                },
            )

        LOGGER.info(
            "Loaded solubility model from %s",
            self.model_path,
        )

        return model

    def predict(
        self,
        inputs: str,
    ) -> SolubilityPrediction:
        scope = (
            self.scope_evaluator.evaluate(
                inputs
            )
        )

        features = self.featurizer.transform(
            [scope.canonical_smiles]
        )

        try:
            raw_predictions = (
                self.model.predict(
                    features
                )
            )
        except Exception as error:
            raise ModelArtifactError(
                "Solubility model inference failed.",
                field="solubility_model",
                context={
                    "path": str(
                        self.model_path
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                },
            ) from error

        predictions = np.asarray(
            raw_predictions,
            dtype=np.float64,
        ).reshape(-1)

        if (
            predictions.shape != (1,)
            or not np.isfinite(
                predictions
            ).all()
        ):
            raise ModelArtifactError(
                "Solubility model returned an invalid prediction.",
                field="solubility_model",
                context={
                    "prediction_shape": list(
                        predictions.shape
                    ),
                },
            )

        applicability_warning = None

        if not scope.in_scope:
            failed = ", ".join(
                scope.failed_rules
            )

            applicability_warning = (
                "The molecule is outside the Day 1 "
                "AqSolDB drug-like applicability scope "
                f"because it failed: {failed}. "
                "The numerical prediction is returned, "
                "but its reliability may be reduced."
            )

        return SolubilityPrediction(
            predicted_log_s=float(
                predictions[0]
            ),
            model_name=(
                "scaffold_xgboost_combined"
            ),
            model_protocol=(
                "scaffold_split"
            ),
            applicability_warning=(
                applicability_warning
            ),
        )