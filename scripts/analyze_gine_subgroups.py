"""Analyze where GINE differs from XGBoost."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data/processed/"
      "solubility_aqsoldb_features.npz"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports/day3/subgroups"
)

PROTOCOLS = {
    "random": {
        "train": (
            "data/splits/random/seed_42/"
            "train.csv"
        ),
        "valid": (
            "data/splits/random/seed_42/"
            "valid.csv"
        ),
        "xgboost": (
            "reports/day2/predictions/"
            "random_xgboost_combined_test.csv"
        ),
    },
    "scaffold": {
        "train": (
            "data/splits/scaffold/seed_42/"
            "train.csv"
        ),
        "valid": (
            "data/splits/scaffold/seed_42/"
            "valid.csv"
        ),
        "xgboost": (
            "reports/day2/predictions/"
            "scaffold_xgboost_combined_test.csv"
        ),
    },
}


def load_feature_cache() -> dict:
    with np.load(
        FEATURE_PATH,
        allow_pickle=False,
    ) as cache:
        return {
            key: cache[key]
            for key in cache.files
        }


def make_index(sample_ids) -> dict:
    return {
        str(sample_id): index
        for index, sample_id
        in enumerate(sample_ids)
    }


def array_to_bit_vector(
    fingerprint: np.ndarray,
):
    bit_vector = DataStructs.ExplicitBitVect(
        int(fingerprint.shape[0])
    )

    for bit_index in np.flatnonzero(
        fingerprint
    ):
        bit_vector.SetBit(int(bit_index))

    return bit_vector


def calculate_nearest_neighbors(
    train_fingerprints: np.ndarray,
    test_fingerprints: np.ndarray,
    train_sample_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    train_bit_vectors = [
        array_to_bit_vector(fingerprint)
        for fingerprint in tqdm(
            train_fingerprints,
            desc="Converting train ECFP",
        )
    ]

    nearest_similarities = []
    nearest_sample_ids = []

    for fingerprint in tqdm(
        test_fingerprints,
        desc="Nearest-neighbor search",
    ):
        test_bit_vector = array_to_bit_vector(
            fingerprint
        )

        similarities = (
            DataStructs.BulkTanimotoSimilarity(
                test_bit_vector,
                train_bit_vectors,
            )
        )

        nearest_index = int(
            np.argmax(similarities)
        )

        nearest_similarities.append(
            float(similarities[nearest_index])
        )
        nearest_sample_ids.append(
            train_sample_ids[nearest_index]
        )

    return (
        np.asarray(
            nearest_similarities,
            dtype=np.float64,
        ),
        nearest_sample_ids,
    )


def calculate_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )


def load_gine_ensemble(
    protocol: str,
) -> pd.DataFrame:
    ensemble = None

    for seed in [1, 2, 3]:
        path = (
            PROJECT_ROOT
            / "reports/day3/predictions/"
              f"{protocol}_gine_"
              f"final_seed_{seed}_test.csv"
        )

        table = pd.read_csv(path)[
            [
                "sample_id",
                "y_true",
                "y_pred",
            ]
        ].copy()

        table = table.rename(
            columns={
                "y_true": f"y_true_seed_{seed}",
                "y_pred": f"gine_seed_{seed}",
            }
        )

        if ensemble is None:
            ensemble = table
        else:
            ensemble = ensemble.merge(
                table,
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )

    if ensemble is None:
        raise RuntimeError(
            "No GINE predictions were loaded."
        )

    true_columns = [
        f"y_true_seed_{seed}"
        for seed in [1, 2, 3]
    ]

    maximum_label_difference = (
        ensemble[true_columns].max(axis=1)
        - ensemble[true_columns].min(axis=1)
    ).max()

    if maximum_label_difference >= 1e-6:
        raise ValueError(
            "GINE seed labels are misaligned."
        )

    ensemble["y_true"] = ensemble[
        "y_true_seed_1"
    ]

    ensemble["gine_prediction"] = ensemble[
        [
            "gine_seed_1",
            "gine_seed_2",
            "gine_seed_3",
        ]
    ].mean(axis=1)

    ensemble["gine_seed_standard_deviation"] = (
        ensemble[
            [
                "gine_seed_1",
                "gine_seed_2",
                "gine_seed_3",
            ]
        ].std(axis=1, ddof=1)
    )

    return ensemble[
        [
            "sample_id",
            "y_true",
            "gine_seed_1",
            "gine_seed_2",
            "gine_seed_3",
            "gine_prediction",
            "gine_seed_standard_deviation",
        ]
    ]


def subgroup_statistics(
    table: pd.DataFrame,
    protocol: str,
    dimension: str,
) -> pd.DataFrame:
    rows = []

    for subgroup, group in table.groupby(
        dimension,
        observed=True,
        dropna=False,
    ):
        gine_absolute_error = np.abs(
            group["gine_prediction"]
            - group["y_true"]
        )
        xgboost_absolute_error = np.abs(
            group["xgboost_prediction"]
            - group["y_true"]
        )

        rows.append({
            "protocol": protocol,
            "dimension": dimension,
            "subgroup": str(subgroup),
            "n_samples": len(group),
            "proportion": (
                len(group) / len(table)
            ),
            "mean_nearest_similarity": (
                group[
                    "nearest_train_similarity"
                ].mean()
            ),
            "gine_mae": (
                gine_absolute_error.mean()
            ),
            "xgboost_mae": (
                xgboost_absolute_error.mean()
            ),
            "gine_minus_xgboost_mae": (
                gine_absolute_error.mean()
                - xgboost_absolute_error.mean()
            ),
            "gine_win_rate": float(
                (
                    gine_absolute_error
                    < xgboost_absolute_error
                ).mean()
            ),
            "mean_gine_seed_std": (
                group[
                    "gine_seed_standard_deviation"
                ].mean()
            ),
        })

    return pd.DataFrame(rows)


def analyze_protocol(
    protocol: str,
    protocol_config: dict,
    cache: dict,
    cache_index: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nAnalyzing {protocol}")

    train = pd.read_csv(
        PROJECT_ROOT
        / protocol_config["train"]
    )
    valid = pd.read_csv(
        PROJECT_ROOT
        / protocol_config["valid"]
    )

    development = pd.concat(
        [train, valid],
        ignore_index=True,
    )

    if development["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate development sample IDs."
        )

    ensemble = load_gine_ensemble(
        protocol
    )

    xgboost = pd.read_csv(
        PROJECT_ROOT
        / protocol_config["xgboost"]
    )[
        [
            "sample_id",
            "y_true",
            "y_pred",
        ]
    ].rename(
        columns={
            "y_true": "xgboost_y_true",
            "y_pred": "xgboost_prediction",
        }
    )

    table = ensemble.merge(
        xgboost,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    maximum_label_difference = np.abs(
        table["y_true"]
        - table["xgboost_y_true"]
    ).max()

    if maximum_label_difference >= 1e-6:
        raise ValueError(
            "GINE and XGBoost labels differ."
        )

    train_indices = np.asarray(
        [
            cache_index[sample_id]
            for sample_id
            in development["sample_id"]
        ],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [
            cache_index[sample_id]
            for sample_id in table["sample_id"]
        ],
        dtype=np.int64,
    )

    train_fingerprints = cache["ecfp"][
        train_indices
    ]
    test_fingerprints = cache["ecfp"][
        test_indices
    ]

    (
        nearest_similarities,
        nearest_sample_ids,
    ) = calculate_nearest_neighbors(
        train_fingerprints=train_fingerprints,
        test_fingerprints=test_fingerprints,
        train_sample_ids=(
            development["sample_id"]
            .astype(str)
            .tolist()
        ),
    )

    table["nearest_train_similarity"] = (
        nearest_similarities
    )
    table["nearest_train_sample_id"] = (
        nearest_sample_ids
    )

    table["smiles"] = cache["smiles"][
        test_indices
    ].astype(str)
    table["in_druglike_scope"] = (
        cache["in_druglike_scope"][
            test_indices
        ]
    )

    descriptor_names = (
        cache["descriptor_names"]
        .astype(str)
        .tolist()
    )

    molecular_weight_index = (
        descriptor_names.index(
            "molecular_weight"
        )
    )

    table["molecular_weight"] = (
        cache["descriptors"][
            test_indices,
            molecular_weight_index,
        ]
    )

    print("Calculating molecular scaffolds")

    train_scaffolds = {
        calculate_scaffold(smiles)
        for smiles in cache["smiles"][
            train_indices
        ].astype(str)
    }

    table["scaffold"] = [
        calculate_scaffold(smiles)
        for smiles in tqdm(
            table["smiles"],
            desc="Test scaffolds",
        )
    ]

    table["scaffold_status"] = [
        (
            "acyclic"
            if scaffold == ""
            else (
                "seen_cyclic_scaffold"
                if scaffold in train_scaffolds
                else "unseen_cyclic_scaffold"
            )
        )
        for scaffold in table["scaffold"]
    ]

    table["similarity_bin"] = pd.cut(
        table["nearest_train_similarity"],
        bins=[
            -np.inf,
            0.30,
            0.50,
            0.70,
            0.85,
            np.inf,
        ],
        labels=[
            "<0.30",
            "0.30-0.50",
            "0.50-0.70",
            "0.70-0.85",
            ">=0.85",
        ],
        right=False,
    )

    table["molecular_weight_bin"] = pd.cut(
        table["molecular_weight"],
        bins=[
            -np.inf,
            200,
            400,
            600,
            np.inf,
        ],
        labels=[
            "<200",
            "200-400",
            "400-600",
            ">=600",
        ],
        right=False,
    )

    table["druglike_status"] = np.where(
        table["in_druglike_scope"],
        "drug_like",
        "outside_drug_like_scope",
    )

    table["gine_absolute_error"] = np.abs(
        table["gine_prediction"]
        - table["y_true"]
    )
    table["xgboost_absolute_error"] = np.abs(
        table["xgboost_prediction"]
        - table["y_true"]
    )
    table["gine_error_advantage"] = (
        table["xgboost_absolute_error"]
        - table["gine_absolute_error"]
    )
    table["better_model"] = np.where(
        table["gine_absolute_error"]
        < table["xgboost_absolute_error"],
        "GINE",
        "XGBoost",
    )

    subgroup_tables = []

    for dimension in [
        "similarity_bin",
        "scaffold_status",
        "molecular_weight_bin",
        "druglike_status",
    ]:
        subgroup_tables.append(
            subgroup_statistics(
                table=table,
                protocol=protocol,
                dimension=dimension,
            )
        )

    subgroup_table = pd.concat(
        subgroup_tables,
        ignore_index=True,
    )

    return table, subgroup_table


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = load_feature_cache()
    cache_index = make_index(
        cache["sample_ids"]
    )

    all_subgroups = []
    protocol_summaries = {}

    for protocol, protocol_config in (
        PROTOCOLS.items()
    ):
        table, subgroups = analyze_protocol(
            protocol=protocol,
            protocol_config=protocol_config,
            cache=cache,
            cache_index=cache_index,
        )

        sample_path = (
            OUTPUT_DIRECTORY
            / f"{protocol}_sample_analysis.csv"
        )
        subgroup_path = (
            OUTPUT_DIRECTORY
            / f"{protocol}_subgroup_metrics.csv"
        )
        advantage_path = (
            OUTPUT_DIRECTORY
            / f"{protocol}_gine_advantages.csv"
        )
        failure_path = (
            OUTPUT_DIRECTORY
            / f"{protocol}_gine_failures.csv"
        )

        table.to_csv(
            sample_path,
            index=False,
        )
        subgroups.to_csv(
            subgroup_path,
            index=False,
        )

        table.sort_values(
            "gine_error_advantage",
            ascending=False,
        ).head(50).to_csv(
            advantage_path,
            index=False,
        )

        table.sort_values(
            "gine_absolute_error",
            ascending=False,
        ).head(50).to_csv(
            failure_path,
            index=False,
        )

        overall_gine_mae = float(
            table["gine_absolute_error"].mean()
        )
        overall_xgboost_mae = float(
            table[
                "xgboost_absolute_error"
            ].mean()
        )

        protocol_summaries[protocol] = {
            "n_samples": len(table),
            "gine_ensemble_mae": (
                overall_gine_mae
            ),
            "xgboost_mae": (
                overall_xgboost_mae
            ),
            "gine_minus_xgboost_mae": (
                overall_gine_mae
                - overall_xgboost_mae
            ),
            "gine_sample_win_rate": float(
                (
                    table["better_model"]
                    == "GINE"
                ).mean()
            ),
            "nearest_similarity_mean": float(
                table[
                    "nearest_train_similarity"
                ].mean()
            ),
            "nearest_similarity_median": float(
                table[
                    "nearest_train_similarity"
                ].median()
            ),
        }

        all_subgroups.append(subgroups)

    combined_subgroups = pd.concat(
        all_subgroups,
        ignore_index=True,
    )

    combined_subgroups.to_csv(
        OUTPUT_DIRECTORY
        / "gine_xgboost_subgroup_metrics.csv",
        index=False,
    )

    with (
        OUTPUT_DIRECTORY
        / "subgroup_analysis_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            protocol_summaries,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nOverall summary")
    print("----------------")

    for protocol, summary in (
        protocol_summaries.items()
    ):
        print(f"\n{protocol}")
        print(
            "GINE ensemble MAE:",
            round(
                summary["gine_ensemble_mae"],
                4,
            ),
        )
        print(
            "XGBoost MAE:",
            round(
                summary["xgboost_mae"],
                4,
            ),
        )
        print(
            "GINE sample win rate:",
            round(
                summary[
                    "gine_sample_win_rate"
                ],
                4,
            ),
        )

    print(
        "\nSaved subgroup analysis to:",
        OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    main()