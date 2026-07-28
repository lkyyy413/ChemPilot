from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import hashlib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class ReactionFingerprintConfig:
    radius: int = 2
    number_of_bits: int = 2048
    include_chirality: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ReactionFingerprintFeaturizer:
    """Generate label-independent reaction fingerprints.

    Components on each side of the reaction are combined
    using bitwise OR. The difference fingerprint is product
    minus reactant and therefore contains values in
    {-1, 0, 1}.
    """

    def __init__(
        self,
        config: ReactionFingerprintConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ReactionFingerprintConfig()
        )

        self.generator = (
            rdFingerprintGenerator.GetMorganGenerator(
                radius=self.config.radius,
                fpSize=self.config.number_of_bits,
                includeChirality=(
                    self.config.include_chirality
                ),
            )
        )

    @staticmethod
    def normalize_smiles_sequence(
        values,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            return [values]

        if isinstance(values, np.ndarray):
            values = values.tolist()

        if isinstance(values, (list, tuple)):
            return [
                str(value)
                for value in values
                if value is not None
                and str(value).strip()
            ]

        return [str(values)]

    def molecule_fingerprint(
        self,
        smiles: str,
    ) -> np.ndarray:
        molecule = Chem.MolFromSmiles(
            smiles
        )

        if molecule is None:
            raise ValueError(
                "Unable to parse SMILES for "
                f"fingerprinting: {smiles!r}"
            )

        fingerprint = (
            self.generator.GetFingerprintAsNumPy(
                molecule
            )
        )

        return fingerprint.astype(
            np.uint8,
            copy=False,
        )

    def side_fingerprint(
        self,
        smiles_values: Iterable[str] | np.ndarray,
    ) -> np.ndarray:
        smiles_list = (
            self.normalize_smiles_sequence(
                smiles_values
            )
        )

        fingerprint = np.zeros(
            self.config.number_of_bits,
            dtype=np.uint8,
        )

        for smiles in smiles_list:
            fingerprint |= (
                self.molecule_fingerprint(
                    smiles
                )
            )

        return fingerprint

    def transform_one(
        self,
        reactant_smiles,
        product_smiles,
    ) -> dict[str, np.ndarray]:
        reactant = self.side_fingerprint(
            reactant_smiles
        )

        product = self.side_fingerprint(
            product_smiles
        )

        difference = (
            product.astype(np.int8)
            - reactant.astype(np.int8)
        )

        concatenated = np.concatenate(
            [
                reactant,
                product,
            ]
        ).astype(
            np.uint8,
            copy=False,
        )

        combined = np.concatenate(
            [
                reactant.astype(np.int8),
                product.astype(np.int8),
                difference,
            ]
        )

        return {
            "reactant": reactant,
            "product": product,
            "difference": difference,
            "concatenated": concatenated,
            "combined": combined,
        }

    def transform(
        self,
        reactant_smiles,
        product_smiles,
    ) -> dict[str, np.ndarray]:
        if len(reactant_smiles) != len(
            product_smiles
        ):
            raise ValueError(
                "Reactant and product collections "
                "must have equal lengths."
            )

        outputs = {
            "reactant": [],
            "product": [],
            "difference": [],
            "concatenated": [],
            "combined": [],
        }

        for reactants, products in zip(
            reactant_smiles,
            product_smiles,
        ):
            features = self.transform_one(
                reactants,
                products,
            )

            for name in outputs:
                outputs[name].append(
                    features[name]
                )

        return {
            name: np.stack(values)
            for name, values in outputs.items()
        }

    @property
    def schema(self) -> dict:
        bits = self.config.number_of_bits

        return {
            "radius": self.config.radius,
            "number_of_bits": bits,
            "include_chirality": (
                self.config.include_chirality
            ),
            "side_aggregation": "bitwise_or",
            "representations": {
                "reactant": bits,
                "product": bits,
                "difference": bits,
                "concatenated": 2 * bits,
                "combined": 3 * bits,
            },
            "difference_values": [
                -1,
                0,
                1,
            ],
        }

@dataclass(frozen=True)
class ConditionFingerprintConfig:
    molecular_radius: int = 2
    molecular_bits: int = 2048
    categorical_hash_bits: int = 4096
    include_chirality: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class ConditionFingerprintFeaturizer:
    """Encode candidate reaction conditions.

    Each chemical role receives a molecular fingerprint
    and a stable hashed categorical representation.
    Missing numeric values remain distinguishable through
    explicit missing indicators.
    """

    ROLE_NAMES = (
        "solvent",
        "catalyst",
        "reagent",
    )

    def __init__(
        self,
        config: ConditionFingerprintConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ConditionFingerprintConfig()
        )

        self.generator = (
            rdFingerprintGenerator.GetMorganGenerator(
                radius=(
                    self.config.molecular_radius
                ),
                fpSize=(
                    self.config.molecular_bits
                ),
                includeChirality=(
                    self.config.include_chirality
                ),
            )
        )

    @staticmethod
    def normalize_labels(
        values,
    ) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            return [values]

        if isinstance(values, np.ndarray):
            values = values.tolist()

        if isinstance(values, (list, tuple)):
            return [
                str(value)
                for value in values
                if value is not None
                and str(value).strip()
            ]

        return [str(values)]

    def molecular_fingerprint(
        self,
        labels,
    ) -> np.ndarray:
        result = np.zeros(
            self.config.molecular_bits,
            dtype=np.uint8,
        )

        for label in self.normalize_labels(
            labels
        ):
            if not label.startswith(
                "SMILES:"
            ):
                continue

            smiles = label.removeprefix(
                "SMILES:"
            )

            molecule = Chem.MolFromSmiles(
                smiles
            )

            if molecule is None:
                raise ValueError(
                    "Unable to parse condition "
                    f"SMILES: {smiles!r}"
                )

            fingerprint = (
                self.generator
                .GetFingerprintAsNumPy(
                    molecule
                )
                .astype(
                    np.uint8,
                    copy=False,
                )
            )

            result |= fingerprint

        return result

    def categorical_fingerprint(
        self,
        labels,
    ) -> np.ndarray:
        result = np.zeros(
            self.config.categorical_hash_bits,
            dtype=np.uint8,
        )

        for label in self.normalize_labels(
            labels
        ):
            if not label.startswith(
                "NAME:"
            ):
                continue

            digest = hashlib.sha256(
                label.encode("utf-8")
            ).digest()

            index = (
                int.from_bytes(
                    digest[:8],
                    byteorder="big",
                    signed=False,
                )
                % self.config.categorical_hash_bits
            )

            result[index] = 1

        return result

    @staticmethod
    def numeric_features(
        temperature_celsius,
        reaction_time_hours,
        solvent_missing: bool,
        catalyst_missing: bool,
        reagent_missing: bool,
    ) -> np.ndarray:
        temperature_missing = bool(
            pd.isna(
                temperature_celsius
            )
        )

        time_missing = bool(
            pd.isna(
                reaction_time_hours
            )
        )

        temperature_value = (
            0.0
            if temperature_missing
            else float(
                temperature_celsius
            )
        )

        time_value = (
            0.0
            if time_missing
            else float(
                reaction_time_hours
            )
        )

        return np.asarray(
            [
                temperature_value,
                float(temperature_missing),
                time_value,
                float(time_missing),
                float(solvent_missing),
                float(catalyst_missing),
                float(reagent_missing),
            ],
            dtype=np.float32,
        )

    def transform_one(
        self,
        solvent_labels,
        catalyst_labels,
        reagent_labels,
        temperature_celsius,
        reaction_time_hours,
    ) -> dict[str, np.ndarray]:
        role_labels = {
            "solvent": (
                self.normalize_labels(
                    solvent_labels
                )
            ),
            "catalyst": (
                self.normalize_labels(
                    catalyst_labels
                )
            ),
            "reagent": (
                self.normalize_labels(
                    reagent_labels
                )
            ),
        }

        outputs = {}

        combined_parts = []

        for role_name in self.ROLE_NAMES:
            labels = role_labels[
                role_name
            ]

            molecular = (
                self.molecular_fingerprint(
                    labels
                )
            )

            categorical = (
                self.categorical_fingerprint(
                    labels
                )
            )

            outputs[
                f"{role_name}_molecular"
            ] = molecular

            outputs[
                f"{role_name}_categorical"
            ] = categorical

            combined_parts.extend(
                [
                    molecular.astype(
                        np.float32
                    ),
                    categorical.astype(
                        np.float32
                    ),
                ]
            )

        numeric = self.numeric_features(
            temperature_celsius=(
                temperature_celsius
            ),
            reaction_time_hours=(
                reaction_time_hours
            ),
            solvent_missing=(
                len(
                    role_labels["solvent"]
                )
                == 0
            ),
            catalyst_missing=(
                len(
                    role_labels["catalyst"]
                )
                == 0
            ),
            reagent_missing=(
                len(
                    role_labels["reagent"]
                )
                == 0
            ),
        )

        outputs["numeric"] = numeric

        combined_parts.append(numeric)

        outputs["combined"] = (
            np.concatenate(
                combined_parts
            ).astype(
                np.float32,
                copy=False,
            )
        )

        return outputs

    @property
    def combined_dimension(self) -> int:
        return (
            len(self.ROLE_NAMES)
            * (
                self.config.molecular_bits
                + self.config.categorical_hash_bits
            )
            + 7
        )

    @property
    def schema(self) -> dict:
        return {
            "roles": list(
                self.ROLE_NAMES
            ),
            "molecular_radius": (
                self.config.molecular_radius
            ),
            "molecular_bits_per_role": (
                self.config.molecular_bits
            ),
            "categorical_hash_bits_per_role": (
                self.config.categorical_hash_bits
            ),
            "categorical_hash": "sha256",
            "combined_dimension": (
                self.combined_dimension
            ),
            "numeric_feature_names": [
                "temperature_celsius_or_zero",
                "temperature_missing",
                "reaction_time_hours_or_zero",
                "reaction_time_missing",
                "solvent_missing",
                "catalyst_missing",
                "reagent_missing",
            ],
        }