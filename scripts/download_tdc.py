"""Download and preserve the raw TDC molecular property dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tdc.single_pred import ADME


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TDC_CACHE_DIR = RAW_DIR / "tdc_cache"

DATASET_NAME = "Solubility_AqSolDB"
OUTPUT_PATH = RAW_DIR / "solubility_aqsoldb_raw.csv"
METADATA_PATH = RAW_DIR / "solubility_aqsoldb_metadata.json"

REQUIRED_COLUMNS = {"Drug_ID", "Drug", "Y"}


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def validate_raw_dataframe(dataframe: pd.DataFrame) -> None:
    """Check whether the downloaded table has the expected columns."""
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Actual columns: {dataframe.columns.tolist()}"
        )

    if dataframe.empty:
        raise ValueError("The downloaded dataset is empty.")


def download_dataset(overwrite: bool = False) -> None:
    """Download the TDC dataset and preserve an immutable raw snapshot."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TDC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not overwrite:
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Raw data are not overwritten by default. "
            "Use --overwrite only when intentionally refreshing the dataset."
        )

    logging.info("Downloading TDC dataset: %s", DATASET_NAME)

    dataset = ADME(
        name=DATASET_NAME,
        path=str(TDC_CACHE_DIR),
    )
    dataframe = dataset.get_data()

    validate_raw_dataframe(dataframe)

    temporary_path = OUTPUT_PATH.with_suffix(".csv.tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(OUTPUT_PATH)

    metadata = {
        "dataset_name": DATASET_NAME,
        "task": "ADME aqueous solubility regression",
        "source": "Therapeutics Data Commons",
        "source_url": "https://tdcommons.ai/single_pred_tasks/adme/",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "number_of_rows": int(len(dataframe)),
        "number_of_columns": int(dataframe.shape[1]),
        "columns": dataframe.columns.tolist(),
        "dtypes": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "missing_values": {
            column: int(value)
            for column, value in dataframe.isna().sum().items()
        },
        "sha256": calculate_sha256(OUTPUT_PATH),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("Saved raw data to: %s", OUTPUT_PATH)
    logging.info("Saved metadata to: %s", METADATA_PATH)
    logging.info("Dataset shape: %s", dataframe.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the raw TDC AqSolDB dataset."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally overwrite an existing raw snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    download_dataset(overwrite=args.overwrite)


if __name__ == "__main__":
    main()