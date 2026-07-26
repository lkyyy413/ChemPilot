"""Convert molecular SMILES into PyTorch Geometric graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem
from torch_geometric.data import Data

from chempilot.features.molecular import smiles_to_mol


MAX_ATOMIC_NUMBER = 118
MAX_DEGREE = 10
MAX_TOTAL_VALENCE = 12
MIN_FORMAL_CHARGE = -5
MAX_FORMAL_CHARGE = 5

HYBRIDIZATION_TO_INDEX = {
    Chem.HybridizationType.UNSPECIFIED: 0,
    Chem.HybridizationType.S: 1,
    Chem.HybridizationType.SP: 2,
    Chem.HybridizationType.SP2: 3,
    Chem.HybridizationType.SP3: 4,
    Chem.HybridizationType.SP3D: 5,
    Chem.HybridizationType.SP3D2: 6,
    Chem.HybridizationType.OTHER: 7,
}

CHIRALITY_TO_INDEX = {
    Chem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.ChiralType.CHI_OTHER: 3,
}

BOND_TYPE_TO_INDEX = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

BOND_STEREO_TO_INDEX = {
    Chem.BondStereo.STEREONONE: 0,
    Chem.BondStereo.STEREOANY: 1,
    Chem.BondStereo.STEREOZ: 2,
    Chem.BondStereo.STEREOE: 3,
    Chem.BondStereo.STEREOCIS: 4,
    Chem.BondStereo.STEREOTRANS: 5,
}

ATOM_FEATURE_NAMES = [
    "atomic_number",
    "degree",
    "total_valence",
    "formal_charge",
    "hybridization",
    "aromatic",
    "chirality",
]

BOND_FEATURE_NAMES = [
    "bond_type",
    "conjugated",
    "in_ring",
    "stereo",
]

ATOM_FEATURE_CARDINALITIES = [
    MAX_ATOMIC_NUMBER + 1,
    MAX_DEGREE + 2,
    MAX_TOTAL_VALENCE + 2,
    (
        MAX_FORMAL_CHARGE
        - MIN_FORMAL_CHARGE
        + 2
    ),
    len(HYBRIDIZATION_TO_INDEX) + 1,
    2,
    len(CHIRALITY_TO_INDEX) + 1,
]

BOND_FEATURE_CARDINALITIES = [
    len(BOND_TYPE_TO_INDEX) + 1,
    2,
    2,
    len(BOND_STEREO_TO_INDEX) + 1,
]


def bounded_index(
    value: int,
    minimum: int,
    maximum: int,
) -> int:
    """Map an integer to a bounded category plus overflow."""

    if minimum <= value <= maximum:
        return value - minimum

    return maximum - minimum + 1


def atomic_number_index(atom: Chem.Atom) -> int:
    atomic_number = atom.GetAtomicNum()

    if 1 <= atomic_number <= MAX_ATOMIC_NUMBER:
        return atomic_number

    return 0


def atom_features(atom: Chem.Atom) -> list[int]:
    formal_charge = bounded_index(
        atom.GetFormalCharge(),
        MIN_FORMAL_CHARGE,
        MAX_FORMAL_CHARGE,
    )

    degree = bounded_index(
        atom.GetDegree(),
        0,
        MAX_DEGREE,
    )

    total_valence = bounded_index(
        atom.GetTotalValence(),
        0,
        MAX_TOTAL_VALENCE,
    )

    hybridization = HYBRIDIZATION_TO_INDEX.get(
        atom.GetHybridization(),
        len(HYBRIDIZATION_TO_INDEX),
    )

    chirality = CHIRALITY_TO_INDEX.get(
        atom.GetChiralTag(),
        len(CHIRALITY_TO_INDEX),
    )

    return [
        atomic_number_index(atom),
        degree,
        total_valence,
        formal_charge,
        hybridization,
        int(atom.GetIsAromatic()),
        chirality,
    ]


def bond_features(bond: Chem.Bond) -> list[int]:
    bond_type = BOND_TYPE_TO_INDEX.get(
        bond.GetBondType(),
        len(BOND_TYPE_TO_INDEX),
    )

    stereo = BOND_STEREO_TO_INDEX.get(
        bond.GetStereo(),
        len(BOND_STEREO_TO_INDEX),
    )

    return [
        bond_type,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        stereo,
    ]


@dataclass
class GraphFeaturizer:
    """Build bidirectional molecular graphs for GINE."""

    def transform_one(
        self,
        smiles: str,
        y: float | None = None,
        sample_id: str | None = None,
        in_druglike_scope: bool | None = None,
    ) -> Data:
        mol = smiles_to_mol(smiles)

        node_features = [
            atom_features(atom)
            for atom in mol.GetAtoms()
        ]

        source_nodes: list[int] = []
        target_nodes: list[int] = []
        edge_features: list[list[int]] = []

        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            features = bond_features(bond)

            source_nodes.extend([begin, end])
            target_nodes.extend([end, begin])
            edge_features.extend([features, features])

        x = torch.tensor(
            node_features,
            dtype=torch.long,
        )

        if source_nodes:
            edge_index = torch.tensor(
                [source_nodes, target_nodes],
                dtype=torch.long,
            )
            edge_attr = torch.tensor(
                edge_features,
                dtype=torch.long,
            )
        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )
            edge_attr = torch.empty(
                (0, len(BOND_FEATURE_NAMES)),
                dtype=torch.long,
            )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=mol.GetNumAtoms(),
        )

        if y is not None:
            data.y = torch.tensor(
                [float(y)],
                dtype=torch.float32,
            )

        if sample_id is not None:
            data.sample_id = str(sample_id)

        data.smiles = str(smiles)

        if in_druglike_scope is not None:
            data.in_druglike_scope = torch.tensor(
                [bool(in_druglike_scope)],
                dtype=torch.bool,
            )

        return data

    @property
    def schema(self) -> dict:
        return {
            "atom_feature_names": ATOM_FEATURE_NAMES,
            "atom_feature_cardinalities": (
                ATOM_FEATURE_CARDINALITIES
            ),
            "bond_feature_names": BOND_FEATURE_NAMES,
            "bond_feature_cardinalities": (
                BOND_FEATURE_CARDINALITIES
            ),
            "directed_edges": True,
        }