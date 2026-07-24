"""Load and align cached molecular features with dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Representation = Literal[
    "descriptors",
    "ecfp",
    "combined",
]


@dataclass
class FeatureBatch:
    """One aligned train, validation, or test feature batch."""

    sample_ids: np.ndarray
    smiles: np.ndarray
    x: np.ndarray
    y: np.ndarray
    in_druglike_scope: np.ndarray
    feature_names: list[str]


class MolecularFeatureStore:
    """Read the feature cache and align it by stable sample ID."""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Feature cache not found: {self.cache_path}"
            )

        with np.load(
            self.cache_path,
            allow_pickle=False,
        ) as cache:
            self.sample_ids = cache["sample_ids"].astype(str)
            self.smiles = cache["smiles"].astype(str)
            self.y = cache["y"].astype(np.float32)
            self.in_druglike_scope = cache[
                "in_druglike_scope"
            ].astype(bool)
            self.descriptors = cache[
                "descriptors"
            ].astype(np.float32)
            self.ecfp = cache["ecfp"].astype(np.uint8)

            self.descriptor_names = (
                cache["descriptor_names"].astype(str).tolist()
            )
            self.ecfp_names = (
                cache["ecfp_names"].astype(str).tolist()
            )

        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(
                "Feature cache contains duplicate sample IDs."
            )

        self.sample_to_index = {
            sample_id: index
            for index, sample_id in enumerate(self.sample_ids)
        }

    def _select_representation(
        self,
        indices: np.ndarray,
        representation: Representation,
    ) -> tuple[np.ndarray, list[str]]:
        if representation == "descriptors":
            return (
                self.descriptors[indices],
                self.descriptor_names.copy(),
            )

        if representation == "ecfp":
            return (
                self.ecfp[indices].astype(np.float32),
                self.ecfp_names.copy(),
            )

        if representation == "combined":
            features = np.concatenate(
                [
                    self.descriptors[indices],
                    self.ecfp[indices].astype(np.float32),
                ],
                axis=1,
            )

            feature_names = (
                self.descriptor_names
                + self.ecfp_names
            )

            return features, feature_names

        raise ValueError(
            f"Unsupported representation: {representation}"
        )

    def load_split(
        self,
        split_path: str | Path,
        representation: Representation,
    ) -> FeatureBatch:
        """Load one CSV split and preserve its exact row order."""

        split_path = Path(split_path)

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        split_df = pd.read_csv(split_path)

        required_columns = {
            "sample_id",
            "smiles_canonical",
            "Y",
            "in_druglike_scope",
        }

        missing_columns = required_columns.difference(
            split_df.columns
        )

        if missing_columns:
            raise ValueError(
                f"{split_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if split_df["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicate sample IDs found in {split_path}"
            )

        missing_sample_ids = [
            sample_id
            for sample_id in split_df["sample_id"].astype(str)
            if sample_id not in self.sample_to_index
        ]

        if missing_sample_ids:
            raise ValueError(
                f"{len(missing_sample_ids)} samples in "
                f"{split_path} are absent from the feature cache. "
                f"Examples: {missing_sample_ids[:5]}"
            )

        indices = np.asarray(
            [
                self.sample_to_index[sample_id]
                for sample_id
                in split_df["sample_id"].astype(str)
            ],
            dtype=np.int64,
        )

        cached_y = self.y[indices]
        split_y = split_df["Y"].to_numpy(dtype=np.float32)

        if not np.allclose(
            cached_y,
            split_y,
            rtol=1e-6,
            atol=1e-6,
        ):
            maximum_difference = float(
                np.max(np.abs(cached_y - split_y))
            )

            raise ValueError(
                "Label mismatch between split and cache. "
                f"Maximum difference: {maximum_difference}"
            )

        cached_smiles = self.smiles[indices]
        split_smiles = (
            split_df["smiles_canonical"]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(cached_smiles, split_smiles):
            raise ValueError(
                "SMILES mismatch between split and feature cache."
            )

        x, feature_names = self._select_representation(
            indices=indices,
            representation=representation,
        )

        return FeatureBatch(
            sample_ids=self.sample_ids[indices],
            smiles=cached_smiles,
            x=x,
            y=cached_y,
            in_druglike_scope=self.in_druglike_scope[indices],
            feature_names=feature_names,
        )