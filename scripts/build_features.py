#!/usr/bin/env python
"""Build a reproducible molecular-feature cache for AqSolDB."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase
from tqdm.auto import tqdm

from chempilot.features.molecular import (
    ECFPFeaturizer,
    RDKitDescriptorFeaturizer,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "solubility_aqsoldb_processed.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "solubility_aqsoldb_features.npz"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "feature_manifest.json"
)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_id",
        "smiles_canonical",
        "Y",
        "in_druglike_scope",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["sample_id"].duplicated().any():
        duplicates = int(df["sample_id"].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicated sample IDs."
        )

    logging.info("Loaded %d processed molecules", len(df))

    smiles = df["smiles_canonical"].astype(str).tolist()

    descriptor_featurizer = RDKitDescriptorFeaturizer()
    ecfp_featurizer = ECFPFeaturizer(
        radius=2,
        n_bits=2048,
        include_chirality=True,
    )

    logging.info("Calculating RDKit descriptors")
    descriptors = descriptor_featurizer.transform(
        tqdm(smiles, desc="RDKit descriptors")
    )

    logging.info("Calculating ECFP4 fingerprints")
    ecfp = ecfp_featurizer.transform(
        tqdm(smiles, desc="ECFP4")
    )

    if descriptors.shape != (len(df), 10):
        raise ValueError(
            f"Unexpected descriptor shape: {descriptors.shape}"
        )

    if ecfp.shape != (len(df), 2048):
        raise ValueError(
            f"Unexpected ECFP shape: {ecfp.shape}"
        )

    if not np.isfinite(descriptors).all():
        raise ValueError(
            "Non-finite values found in RDKit descriptors."
        )

    if not np.isin(ecfp, [0, 1]).all():
        raise ValueError(
            "ECFP contains values other than zero and one."
        )

    zero_fingerprint_rows = int(
        (ecfp.sum(axis=1) == 0).sum()
    )

    np.savez_compressed(
       OUTPUT_PATH,
       sample_ids=np.asarray(
           df["sample_id"].astype(str).tolist(),
           dtype=np.str_,
       ),
       smiles=np.asarray(
           df["smiles_canonical"].astype(str).tolist(),
           dtype=np.str_,
       ),
       y=df["Y"].to_numpy(dtype=np.float32),
       in_druglike_scope=df[
           "in_druglike_scope"
       ].to_numpy(dtype=bool),
       descriptors=descriptors.astype(np.float32),
       ecfp=ecfp.astype(np.uint8),
       descriptor_names=np.asarray(
           descriptor_featurizer.feature_names,
           dtype=np.str_,
       ),
       ecfp_names=np.asarray(
           ecfp_featurizer.feature_names,
           dtype=np.str_,
       ),
    )

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "input_sha256": sha256_file(INPUT_PATH),
        "output_sha256": sha256_file(OUTPUT_PATH),
        "number_of_samples": len(df),
        "descriptor_shape": list(descriptors.shape),
        "descriptor_dtype": str(descriptors.dtype),
        "descriptor_names": (
            descriptor_featurizer.feature_names
        ),
        "ecfp_shape": list(ecfp.shape),
        "ecfp_dtype": str(ecfp.dtype),
        "ecfp_configuration": {
            "name": "Morgan/ECFP4",
            "radius": 2,
            "diameter": 4,
            "n_bits": 2048,
            "include_chirality": True,
            "use_counts": False,
        },
        "zero_fingerprint_rows": zero_fingerprint_rows,
        "chemical_changes": {
            "performed_during_featurization": False,
            "input_column": "smiles_canonical",
        },
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logging.info(
        "Saved feature cache to %s",
        OUTPUT_PATH,
    )
    logging.info(
        "Saved feature manifest to %s",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()