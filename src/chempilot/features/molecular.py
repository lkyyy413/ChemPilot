"""Molecular feature extraction using RDKit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


DESCRIPTOR_NAMES = [
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


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Convert one SMILES string into a sanitized RDKit molecule."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

    return mol


@dataclass
class RDKitDescriptorFeaturizer:
    """Calculate interpretable global molecular descriptors."""

    dtype: type = np.float32

    @property
    def feature_names(self) -> list[str]:
        return DESCRIPTOR_NAMES.copy()

    def transform_one(self, smiles: str) -> np.ndarray:
        mol = smiles_to_mol(smiles)

        values = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Lipinski.HeavyAtomCount(mol),
            Chem.GetFormalCharge(mol),
        ]

        return np.asarray(values, dtype=self.dtype)

    def transform(self, smiles_list: Sequence[str]) -> np.ndarray:
        features = [
            self.transform_one(smiles)
            for smiles in smiles_list
        ]
        return np.vstack(features)


@dataclass
class ECFPFeaturizer:
    """Calculate ECFP4/Morgan binary fingerprints."""

    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True
    dtype: type = np.uint8

    def __post_init__(self) -> None:
        self.generator = GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=self.include_chirality,
        )

    @property
    def feature_names(self) -> list[str]:
        return [
            f"ecfp_{index:04d}"
            for index in range(self.n_bits)
        ]

    def transform_one(self, smiles: str) -> np.ndarray:
        mol = smiles_to_mol(smiles)
        fingerprint = self.generator.GetFingerprintAsNumPy(mol)
        return np.asarray(fingerprint, dtype=self.dtype)

    def transform(self, smiles_list: Sequence[str]) -> np.ndarray:
        features = [
            self.transform_one(smiles)
            for smiles in smiles_list
        ]
        return np.vstack(features)


@dataclass
class CombinedFeaturizer:
    """Concatenate global RDKit descriptors and ECFP features."""

    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True

    def __post_init__(self) -> None:
        self.descriptor_featurizer = RDKitDescriptorFeaturizer()
        self.ecfp_featurizer = ECFPFeaturizer(
            radius=self.radius,
            n_bits=self.n_bits,
            include_chirality=self.include_chirality,
        )

    @property
    def feature_names(self) -> list[str]:
        return (
            self.descriptor_featurizer.feature_names
            + self.ecfp_featurizer.feature_names
        )

    def transform(self, smiles_list: Sequence[str]) -> np.ndarray:
        descriptors = self.descriptor_featurizer.transform(smiles_list)
        ecfp = self.ecfp_featurizer.transform(smiles_list)

        return np.concatenate(
            [descriptors, ecfp.astype(np.float32)],
            axis=1,
        )