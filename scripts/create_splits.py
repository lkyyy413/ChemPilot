"""Create diagnostic random splits and official TDC scaffold splits."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from tdc.benchmark_group import admet_group


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "solubility_aqsoldb_processed.csv"
)
BENCHMARK_CACHE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tdc_benchmark_cache"
)
RANDOM_DIR = PROJECT_ROOT / "data" / "splits" / "random"
SCAFFOLD_DIR = PROJECT_ROOT / "data" / "splits" / "scaffold"
REPORT_PATH = PROJECT_ROOT / "reports" / "split_audit.json"

OUTPUT_COLUMNS = [
    "sample_id",
    "Drug_ID",
    "smiles_canonical",
    "Y",
    "in_druglike_scope",
]


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid benchmark SMILES: {smiles}")

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def calculate_scaffold(smiles: str) -> str:
    """Calculate an achiral Bemis-Murcko scaffold."""
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles,
        includeChirality=False,
    )


def attach_processed_metadata(
    split_dataframe: pd.DataFrame,
    processed: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Map a TDC split back to stable local sample IDs."""
    split = split_dataframe.copy()
    split["smiles_canonical"] = split["Drug"].map(
        canonicalize_smiles
    )

    lookup = processed[
        [
            "sample_id",
            "smiles_canonical",
            "Y",
            "in_druglike_scope",
        ]
    ].rename(columns={"Y": "processed_Y"})

    merged = split.merge(
        lookup,
        on="smiles_canonical",
        how="left",
        validate="one_to_one",
    )

    if merged["sample_id"].isna().any():
        missing = int(merged["sample_id"].isna().sum())
        raise ValueError(
            f"{split_name}: {missing} benchmark molecules "
            "could not be mapped to processed data."
        )

    if not np.allclose(
        merged["Y"].to_numpy(),
        merged["processed_Y"].to_numpy(),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            f"{split_name}: benchmark and processed labels differ."
        )

    return merged[OUTPUT_COLUMNS].copy()


def create_random_splits(
    processed: pd.DataFrame,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Create a label-stratified 70/10/20 diagnostic random split."""
    dataframe = processed.copy()

    # Regression labels are binned only for stratification.
    dataframe["_label_bin"] = pd.qcut(
        dataframe["Y"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    train_valid, test = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=seed,
        stratify=dataframe["_label_bin"],
    )

    train, valid = train_test_split(
        train_valid,
        test_size=0.125,
        random_state=seed,
        stratify=train_valid["_label_bin"],
    )

    return {
        "train": train[OUTPUT_COLUMNS].reset_index(drop=True),
        "valid": valid[OUTPUT_COLUMNS].reset_index(drop=True),
        "test": test[OUTPUT_COLUMNS].reset_index(drop=True),
    }


def id_overlap(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> int:
    return len(
        set(first["sample_id"]).intersection(second["sample_id"])
    )


def scaffold_set(dataframe: pd.DataFrame) -> set[str]:
    return set(
        dataframe["smiles_canonical"].map(calculate_scaffold)
    )


def summarize_split(dataframe: pd.DataFrame) -> dict[str, Any]:
    scaffolds = dataframe["smiles_canonical"].map(
        calculate_scaffold
    )

    return {
        "rows": int(len(dataframe)),
        "druglike_rows": int(
            dataframe["in_druglike_scope"].sum()
        ),
        "label_mean": float(dataframe["Y"].mean()),
        "label_median": float(dataframe["Y"].median()),
        "label_standard_deviation": float(
            dataframe["Y"].std()
        ),
        "unique_scaffolds_including_empty": int(
            scaffolds.nunique()
        ),
        "empty_scaffold_rows": int((scaffolds == "").sum()),
    }


def sample_id_hash(dataframe: pd.DataFrame) -> str:
    ordered_ids = "\n".join(sorted(dataframe["sample_id"]))
    return hashlib.sha256(ordered_ids.encode("utf-8")).hexdigest()


def audit_three_way_split(
    splits: dict[str, pd.DataFrame],
    expected_total: int,
) -> dict[str, Any]:
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    train_scaffolds = scaffold_set(train)
    valid_scaffolds = scaffold_set(valid)
    test_scaffolds = scaffold_set(test)

    return {
        "train": summarize_split(train),
        "valid": summarize_split(valid),
        "test": summarize_split(test),
        "total_rows": int(
            len(train) + len(valid) + len(test)
        ),
        "expected_total_rows": int(expected_total),
        "sample_id_overlaps": {
            "train_valid": id_overlap(train, valid),
            "train_test": id_overlap(train, test),
            "valid_test": id_overlap(valid, test),
        },
        "scaffold_overlaps_including_empty": {
            "train_valid": len(
                train_scaffolds.intersection(valid_scaffolds)
            ),
            "train_test": len(
                train_scaffolds.intersection(test_scaffolds)
            ),
            "valid_test": len(
                valid_scaffolds.intersection(test_scaffolds)
            ),
        },
        "sample_id_hashes": {
            "train": sample_id_hash(train),
            "valid": sample_id_hash(valid),
            "test": sample_id_hash(test),
        },
    }


def save_splits(
    splits: dict[str, pd.DataFrame],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for split_name, dataframe in splits.items():
        dataframe.to_csv(
            output_directory / f"{split_name}.csv",
            index=False,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    config = load_config()
    split_config = config["splits"]

    processed = pd.read_csv(PROCESSED_PATH)

    random_seed = split_config["random_seed"]
    random_splits = create_random_splits(
        processed=processed,
        seed=random_seed,
    )
    save_splits(
        random_splits,
        RANDOM_DIR / f"seed_{random_seed}",
    )

    logging.info("Loading official TDC ADMET benchmark")
    group = admet_group(path=str(BENCHMARK_CACHE))
    benchmark = group.get("Solubility_AqSolDB")

    fixed_test = attach_processed_metadata(
        benchmark["test"],
        processed,
        "official_test",
    )

    SCAFFOLD_DIR.mkdir(parents=True, exist_ok=True)
    fixed_test.to_csv(
        SCAFFOLD_DIR / "test.csv",
        index=False,
    )

    official_seed_reports: dict[str, Any] = {}

    requested_seeds = sorted(
        set(
            split_config["official_final_seeds"]
            + [split_config["official_development_seed"]]
        )
    )

    for seed in requested_seeds:
        logging.info(
            "Creating official train/valid split for seed=%d",
            seed,
        )

        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=benchmark["name"],
            split_type="default",
            seed=seed,
        )

        splits = {
            "train": attach_processed_metadata(
                train_raw,
                processed,
                f"official_train_seed_{seed}",
            ),
            "valid": attach_processed_metadata(
                valid_raw,
                processed,
                f"official_valid_seed_{seed}",
            ),
            "test": fixed_test,
        }

        seed_directory = SCAFFOLD_DIR / f"seed_{seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)

        splits["train"].to_csv(
            seed_directory / "train.csv",
            index=False,
        )
        splits["valid"].to_csv(
            seed_directory / "valid.csv",
            index=False,
        )

        official_seed_reports[str(seed)] = (
            audit_three_way_split(
                splits=splits,
                expected_total=len(processed),
            )
        )

    report = {
        "dataset": "Solubility_AqSolDB",
        "random_split": {
            "purpose": (
                "Diagnostic only; not comparable to the "
                "official TDC leaderboard."
            ),
            "seed": random_seed,
            "audit": audit_three_way_split(
                random_splits,
                expected_total=len(processed),
            ),
        },
        "official_scaffold_split": {
            "purpose": (
                "Primary evaluation protocol compatible "
                "with the TDC ADMET benchmark."
            ),
            "fixed_test_rows": int(len(fixed_test)),
            "seeds": official_seed_reports,
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("Saved split audit to %s", REPORT_PATH)


if __name__ == "__main__":
    main()