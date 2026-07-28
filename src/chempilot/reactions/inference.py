"""Inference interface for reaction conditions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from chempilot.evaluation.applicability import (
    nearest_binary_tanimoto,
)
from chempilot.reactions.features import (
    ReactionFingerprintFeaturizer,
)


DEFAULT_MODEL_ROOT = Path(
    "artifacts/models/day4"
)

DEFAULT_TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

DEFAULT_FEATURE_PATH = Path(
    "data/processed/reactions/features/"
    "reaction_combined.npz"
)

DEFAULT_AD_REPORT_PATH = Path(
    "reports/day4/applicability/"
    "applicability_summary.json"
)

BINARY_STRUCTURE_DIMENSION = 4096
REACTION_FEATURE_DIMENSION = 6144
SUPPORTED_PROTOCOLS = {
    "transformation",
    "reaction_center",
}
SUPPORTED_TARGETS = (
    "solvent",
    "catalyst",
)


class ReactionConditionPredictor:
    """Predict solvent and catalyst condition labels."""

    def __init__(
        self,
        protocol: str = "reaction_center",
        *,
        model_root: str | Path = (
            DEFAULT_MODEL_ROOT
        ),
        target_root: str | Path = (
            DEFAULT_TARGET_ROOT
        ),
        feature_path: str | Path = (
            DEFAULT_FEATURE_PATH
        ),
        ad_report_path: str | Path = (
            DEFAULT_AD_REPORT_PATH
        ),
    ) -> None:
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                "protocol must be one of "
                f"{sorted(SUPPORTED_PROTOCOLS)}."
            )

        self.protocol = protocol
        self.model_root = Path(model_root)
        self.target_root = Path(target_root)
        self.feature_path = Path(
            feature_path
        )
        self.ad_report_path = Path(
            ad_report_path
        )

        self.featurizer = (
            ReactionFingerprintFeaturizer()
        )

        self.artifacts = {}
        self.reference_features = {}
        self.reference_metadata = {}
        self.ad_thresholds = {}

        self._load_resources()

    def _load_resources(self) -> None:
        reaction_cache = sparse.load_npz(
            self.feature_path
        ).tocsr()

        if (
            reaction_cache.shape[1]
            != REACTION_FEATURE_DIMENSION
        ):
            raise ValueError(
                "Unexpected reaction feature "
                f"dimension: {reaction_cache.shape[1]}."
            )

        with self.ad_report_path.open(
            encoding="utf-8"
        ) as file:
            ad_report = json.load(file)

        for target in SUPPORTED_TARGETS:
            artifact_path = (
                self.model_root
                / self.protocol
                / (
                    f"{target}_logistic_"
                    "final.joblib"
                )
            )

            artifact = joblib.load(
                artifact_path
            )

            if (
                artifact["protocol"]
                != self.protocol
            ):
                raise ValueError(
                    "Model protocol mismatch in "
                    f"{artifact_path}."
                )

            if artifact["target"] != target:
                raise ValueError(
                    "Model target mismatch in "
                    f"{artifact_path}."
                )

            if (
                artifact[
                    "feature_representation"
                ]
                != "reaction_combined"
            ):
                raise ValueError(
                    "Unsupported model feature "
                    "representation."
                )

            model = artifact["model"]
            vocabulary = list(
                artifact["vocabulary"]
            )

            if (
                model.n_features_in_
                != REACTION_FEATURE_DIMENSION
            ):
                raise ValueError(
                    "Model input dimension "
                    "does not match the "
                    "reaction featurizer."
                )

            samples_path = (
                self.target_root
                / self.protocol
                / target
                / "samples.parquet"
            )

            samples = pd.read_parquet(
                samples_path
            )

            training_mask = (
                samples["split"].isin(
                    ["train", "valid"]
                )
                & samples[
                    "has_any_known_target"
                ]
            )

            reference_metadata = (
                samples.loc[
                    training_mask,
                    [
                        "feature_row_index",
                        "transformation_signature",
                        "reaction_center_signature",
                        "reaction_type",
                    ],
                ]
                .reset_index(drop=True)
            )

            feature_indices = (
                reference_metadata[
                    "feature_row_index"
                ].to_numpy(
                    dtype=np.int64
                )
            )

            reference_features = (
                reaction_cache[
                    feature_indices,
                    :BINARY_STRUCTURE_DIMENSION,
                ]
            )

            task_key = (
                f"{self.protocol}|{target}"
            )

            ad_threshold = float(
                ad_report[
                    "tasks"
                ][task_key][
                    "ad_threshold"
                ]
            )

            self.artifacts[target] = {
                "model": model,
                "vocabulary": vocabulary,
                "regularization_c": float(
                    artifact[
                        "regularization_c"
                    ]
                ),
            }

            self.reference_features[
                target
            ] = reference_features

            self.reference_metadata[
                target
            ] = reference_metadata

            self.ad_thresholds[
                target
            ] = ad_threshold

    @staticmethod
    def _normalize_smiles_input(
        value,
        field_name: str,
    ) -> list[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(
            value,
            (list, tuple),
        ):
            values = list(value)
        elif isinstance(
            value,
            np.ndarray,
        ):
            if value.ndim == 0:
                values = [
                    value.item()
                ]
            else:
                values = (
                    value.reshape(-1).tolist()
                )

        elif isinstance(
            value,
            pd.Series,
        ):
            values = value.tolist()
        else:
            raise TypeError(
                f"{field_name} must be a "
                "SMILES string or a sequence "
                "of SMILES strings; received "
                f"{type(value).__name__}."
            )

        normalized = [
            str(item).strip()
            for item in values
            if item is not None
            and str(item).strip()
        ]

        if not normalized:
            raise ValueError(
                f"{field_name} is empty."
            )

        return normalized

    def _predict_target(
        self,
        *,
        target: str,
        combined_features: sparse.csr_matrix,
        binary_features: sparse.csr_matrix,
        top_k: int,
    ) -> dict:
        artifact = self.artifacts[target]
        model = artifact["model"]
        vocabulary = artifact["vocabulary"]

        probabilities = np.asarray(
            model.predict_proba(
                combined_features
            )
        )

        if probabilities.shape != (
            1,
            len(vocabulary),
        ):
            raise RuntimeError(
                "Unexpected probability shape."
            )

        scores = probabilities[0]

        number_to_return = min(
            top_k,
            len(vocabulary),
        )

        top_indices = np.argsort(
            scores
        )[::-1][
            :number_to_return
        ]

        recommendations = [
            {
                "rank": rank,
                "label": vocabulary[index],
                "score": float(
                    scores[index]
                ),
            }
            for rank, index in enumerate(
                top_indices,
                start=1,
            )
        ]

        (
            nearest_similarities,
            nearest_indices,
        ) = nearest_binary_tanimoto(
            binary_features,
            self.reference_features[
                target
            ],
        )

        nearest_index = int(
            nearest_indices[0]
        )

        nearest_similarity = float(
            nearest_similarities[0]
        )

        nearest = (
            self.reference_metadata[
                target
            ]
            .iloc[nearest_index]
        )

        threshold = self.ad_thresholds[
            target
        ]

        return {
            "target": target,
            "top_k": recommendations,
            "score_interpretation": (
                "Uncalibrated ranking score; "
                "not a calibrated probability."
            ),
            "applicability_domain": {
                "in_domain": bool(
                    nearest_similarity
                    >= threshold
                ),
                "nearest_similarity": (
                    nearest_similarity
                ),
                "threshold": threshold,
                "threshold_definition": (
                    "5th percentile of final "
                    "training-set leave-one-out "
                    "nearest-neighbor Tanimoto "
                    "similarity"
                ),
                "nearest_train_transformation": (
                    nearest[
                        "transformation_signature"
                    ]
                ),
                "nearest_train_reaction_center": (
                    nearest[
                        "reaction_center_signature"
                    ]
                ),
                "nearest_train_reaction_type": (
                    nearest["reaction_type"]
                ),
            },
        }

    def predict(
        self,
        reactant_smiles,
        product_smiles,
        *,
        top_k: int = 5,
    ) -> dict:
        """Predict Top-K solvents and catalysts."""

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        reactants = (
            self._normalize_smiles_input(
                reactant_smiles,
                "reactant_smiles",
            )
        )

        products = (
            self._normalize_smiles_input(
                product_smiles,
                "product_smiles",
            )
        )

        features = (
            self.featurizer.transform_one(
                reactants,
                products,
            )
        )

        combined_array = np.asarray(
            features["combined"]
        )

        if combined_array.ndim == 1:
            combined_array = (
                combined_array.reshape(
                    1,
                    -1,
                )
            )

        if combined_array.shape != (
            1,
            REACTION_FEATURE_DIMENSION,
        ):
            raise RuntimeError(
                "Unexpected generated feature "
                f"shape: {combined_array.shape}."
            )

        combined_features = (
            sparse.csr_matrix(
                combined_array,
                dtype=np.float32,
            )
        )

        binary_features = (
            combined_features[
                :,
                :BINARY_STRUCTURE_DIMENSION,
            ]
        )

        predictions = {
            target: self._predict_target(
                target=target,
                combined_features=(
                    combined_features
                ),
                binary_features=(
                    binary_features
                ),
                top_k=top_k,
            )
            for target in SUPPORTED_TARGETS
        }

        return {
            "protocol": self.protocol,
            "input": {
                "reactant_smiles": reactants,
                "product_smiles": products,
            },
            "predictions": predictions,
            "interpretation": {
                "solvent_and_catalyst_are_independent": (
                    True
                ),
                "joint_condition_ranking_available": (
                    False
                ),
                "applicability_note": (
                    "in_domain indicates structural "
                    "coverage relative to training "
                    "reactions; it does not guarantee "
                    "prediction correctness."
                ),
            },
        }