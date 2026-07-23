"""Audit the raw AqSolDB snapshot without modifying or deleting samples."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "solubility_aqsoldb_raw.csv"
FLAGS_PATH = PROJECT_ROOT / "data" / "interim" / "raw_audit_flags.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "raw_data_audit.json"

# 常见药物样有机分子中允许的元素。
# 这不是绝对化学规则，只用于发现需要人工检查的样本。
COMMON_DRUGLIKE_ATOMIC_NUMBERS = {
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


def inspect_smiles(smiles: Any) -> dict[str, Any]:
    """Parse one SMILES string and calculate non-destructive audit flags."""
    if not isinstance(smiles, str) or not smiles.strip():
        return {
            "is_valid_smiles": False,
            "canonical_smiles": None,
            "num_fragments": np.nan,
            "is_multifragment": False,
            "formal_charge": np.nan,
            "heavy_atom_count": np.nan,
            "molecular_weight": np.nan,
            "elements": None,
            "has_uncommon_element": False,
        }

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return {
            "is_valid_smiles": False,
            "canonical_smiles": None,
            "num_fragments": np.nan,
            "is_multifragment": False,
            "formal_charge": np.nan,
            "heavy_atom_count": np.nan,
            "molecular_weight": np.nan,
            "elements": None,
            "has_uncommon_element": False,
        }

    atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
    element_symbols = sorted(
        {atom.GetSymbol() for atom in molecule.GetAtoms()}
    )
    num_fragments = len(Chem.GetMolFrags(molecule))
    formal_charge = sum(
        atom.GetFormalCharge() for atom in molecule.GetAtoms()
    )

    return {
        "is_valid_smiles": True,
        "canonical_smiles": Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        ),
        "num_fragments": num_fragments,
        "is_multifragment": num_fragments > 1,
        "formal_charge": formal_charge,
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "molecular_weight": Descriptors.MolWt(molecule),
        "elements": ",".join(element_symbols),
        "has_uncommon_element": not atomic_numbers.issubset(
            COMMON_DRUGLIKE_ATOMIC_NUMBERS
        ),
    }


def build_report(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Summarize data-quality findings."""
    valid = dataframe[dataframe["is_valid_smiles"]].copy()

    canonical_group_sizes = valid.groupby("canonical_smiles").size()
    duplicate_canonical_groups = canonical_group_sizes[
        canonical_group_sizes > 1
    ]

    label_ranges = valid.groupby("canonical_smiles")["Y"].agg(
        lambda values: float(values.max() - values.min())
    )
    conflicting_label_groups = label_ranges[label_ranges > 1e-8]

    return {
        "dataset": "Solubility_AqSolDB",
        "label_name": "LogS",
        "label_unit": "log10(mol/L)",
        "total_rows": int(len(dataframe)),
        "missing_drug_id": int(dataframe["Drug_ID"].isna().sum()),
        "missing_smiles": int(dataframe["Drug"].isna().sum()),
        "missing_label": int(dataframe["Y"].isna().sum()),
        "invalid_smiles": int((~dataframe["is_valid_smiles"]).sum()),
        "valid_smiles": int(dataframe["is_valid_smiles"].sum()),
        "raw_smiles_duplicate_rows": int(
            dataframe["Drug"].duplicated(keep=False).sum()
        ),
        "canonical_duplicate_groups": int(
            len(duplicate_canonical_groups)
        ),
        "canonical_duplicate_rows": int(
            duplicate_canonical_groups.sum()
        ),
        "conflicting_label_groups": int(
            len(conflicting_label_groups)
        ),
        "multifragment_molecules": int(
            dataframe["is_multifragment"].sum()
        ),
        "nonzero_formal_charge": int(
            (dataframe["formal_charge"].fillna(0) != 0).sum()
        ),
        "uncommon_element_molecules": int(
            dataframe["has_uncommon_element"].sum()
        ),
        "label_statistics": {
            "minimum": float(dataframe["Y"].min()),
            "maximum": float(dataframe["Y"].max()),
            "mean": float(dataframe["Y"].mean()),
            "median": float(dataframe["Y"].median()),
            "standard_deviation": float(dataframe["Y"].std()),
        },
        "molecular_weight_statistics_valid_only": {
            "minimum": float(valid["molecular_weight"].min()),
            "maximum": float(valid["molecular_weight"].max()),
            "median": float(valid["molecular_weight"].median()),
        },
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_PATH}"
        )

    # 避免无效SMILES产生大量RDKit错误输出，数量仍会被记录。
    RDLogger.DisableLog("rdApp.error")

    dataframe = pd.read_csv(INPUT_PATH)
    logging.info("Loaded %d raw rows", len(dataframe))

    audit_columns = dataframe["Drug"].apply(inspect_smiles).apply(pd.Series)
    audited_dataframe = pd.concat([dataframe, audit_columns], axis=1)

    FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    audited_dataframe.to_csv(FLAGS_PATH, index=False)

    report = build_report(audited_dataframe)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("Saved audit flags to %s", FLAGS_PATH)
    logging.info("Saved audit report to %s", REPORT_PATH)


if __name__ == "__main__":
    main()