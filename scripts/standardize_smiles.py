"""Conservatively standardize AqSolDB without changing chemical composition."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem, RDLogger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def canonicalize_smiles(smiles: str) -> str:
    """Create canonical isomeric SMILES while preserving all fragments."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES encountered: {smiles}")

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def contains_carbon(smiles: str) -> bool:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return False

    return any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    config = load_config()
    paths = config["paths"]
    scope = config["druglike_scope"]

    audit_path = PROJECT_ROOT / paths["audit_flags"]
    output_path = PROJECT_ROOT / paths["processed_data"]
    report_path = PROJECT_ROOT / paths["processing_report"]

    dataframe = pd.read_csv(audit_path)
    logging.info("Loaded %d audited rows", len(dataframe))

    dataframe["smiles_original"] = dataframe["Drug"]
    dataframe["smiles_canonical"] = dataframe["Drug"].map(
        canonicalize_smiles
    )
    dataframe["contains_carbon"] = dataframe[
        "smiles_canonical"
    ].map(contains_carbon)

    dataframe["passes_fragment_rule"] = (
        dataframe["num_fragments"] == 1
    )
    dataframe["passes_element_rule"] = (
        ~dataframe["has_uncommon_element"]
    )
    dataframe["passes_molecular_weight_rule"] = dataframe[
        "molecular_weight"
    ].between(
        scope["minimum_molecular_weight"],
        scope["maximum_molecular_weight"],
        inclusive="both",
    )

    dataframe["in_druglike_scope"] = (
        dataframe["is_valid_smiles"]
        & dataframe["passes_fragment_rule"]
        & dataframe["passes_element_rule"]
        & dataframe["passes_molecular_weight_rule"]
        & dataframe["contains_carbon"]
    )

    dataframe["sample_id"] = [
        f"AQSOL_{index:05d}"
        for index in range(len(dataframe))
    ]

    output_columns = [
        "sample_id",
        "Drug_ID",
        "smiles_original",
        "smiles_canonical",
        "Y",
        "is_valid_smiles",
        "num_fragments",
        "is_multifragment",
        "formal_charge",
        "heavy_atom_count",
        "molecular_weight",
        "elements",
        "has_uncommon_element",
        "contains_carbon",
        "passes_fragment_rule",
        "passes_element_rule",
        "passes_molecular_weight_rule",
        "in_druglike_scope",
    ]

    processed = dataframe[output_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    processed.to_csv(output_path, index=False)

    report = {
        "input_rows": int(len(processed)),
        "output_rows": int(len(processed)),
        "rows_removed": 0,
        "canonical_smiles_changed": int(
            (
                processed["smiles_original"]
                != processed["smiles_canonical"]
            ).sum()
        ),
        "official_benchmark_scope": int(len(processed)),
        "druglike_scope": int(
            processed["in_druglike_scope"].sum()
        ),
        "outside_druglike_scope": int(
            (~processed["in_druglike_scope"]).sum()
        ),
        "failed_fragment_rule": int(
            (~processed["passes_fragment_rule"]).sum()
        ),
        "failed_element_rule": int(
            (~processed["passes_element_rule"]).sum()
        ),
        "failed_molecular_weight_rule": int(
            (~processed["passes_molecular_weight_rule"]).sum()
        ),
        "failed_carbon_rule": int(
            (~processed["contains_carbon"]).sum()
        ),
        "chemical_changes": {
            "fragments_removed": False,
            "charges_neutralized": False,
            "stereochemistry_preserved": True,
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("Saved processed data to %s", output_path)
    logging.info(
        "Drug-like scope: %d / %d",
        report["druglike_scope"],
        report["input_rows"],
    )


if __name__ == "__main__":
    main()