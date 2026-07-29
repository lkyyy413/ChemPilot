"""Select RXNFP pooling for reaction retrieval using validation data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from chempilot.reactions.retrieval import (
    ReactionEmbeddingRetriever,
)


PROTOCOLS = (
    "transformation",
    "reaction_center",
)

POOLING_METHODS = (
    "cls",
    "masked_mean",
)

TARGETS = (
    "solvent",
    "catalyst",
)

TOP_K_VALUES = (
    1,
    5,
    10,
)

FEATURE_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_embeddings.npz"
)

METADATA_PATH = Path(
    "data/processed/reactions/"
    "features/day5/"
    "rxnfp_reaction_metadata.parquet"
)

TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

REPORT_ROOT = Path(
    "reports/day5/retrieval"
)

REPORT_PATH = (
    REPORT_ROOT
    / "retrieval_validation.json"
)

NEIGHBOR_PATH = (
    REPORT_ROOT
    / "retrieval_validation_neighbors.csv"
)


def reciprocal_rank_at_k(
    matches,
) -> float:
    positions = np.flatnonzero(
        matches
    )

    if len(positions) == 0:
        return 0.0

    return 1.0 / (
        int(positions[0]) + 1
    )


def condition_retrieval_metrics(
    protocol,
    target,
    query_feature_indices,
    neighbor_feature_indices,
    metadata,
):
    target_directory = (
        TARGET_ROOT
        / protocol
        / target
    )

    samples = pd.read_parquet(
        target_directory
        / "samples.parquet"
    )

    targets = sparse.load_npz(
        target_directory
        / "targets.npz"
    ).tocsr()

    row_by_transformation = (
        pd.Series(
            np.arange(
                len(samples),
                dtype=np.int64,
            ),
            index=samples[
                "transformation_signature"
            ],
        )
    )

    query_signatures = metadata.iloc[
        query_feature_indices
    ][
        "transformation_signature"
    ]

    query_target_rows = (
        query_signatures.map(
            row_by_transformation
        ).to_numpy(
            dtype=np.int64
        )
    )

    neighbor_signatures = (
        metadata.iloc[
            neighbor_feature_indices
            .reshape(-1)
        ][
            "transformation_signature"
        ]
        .to_numpy()
        .reshape(
            neighbor_feature_indices.shape
        )
    )

    neighbor_target_rows = np.empty(
        neighbor_signatures.shape,
        dtype=np.int64,
    )

    for row_index in range(
        len(neighbor_signatures)
    ):
        neighbor_target_rows[
            row_index
        ] = (
            pd.Series(
                neighbor_signatures[
                    row_index
                ]
            )
            .map(
                row_by_transformation
            )
            .to_numpy(
                dtype=np.int64
            )
        )

    results = {}

    for k in TOP_K_VALUES:
        recalls = []
        hits = []

        for query_index, target_row in (
            enumerate(query_target_rows)
        ):
            query_labels = (
                targets[
                    target_row
                ].toarray()
                .reshape(-1)
                .astype(bool)
            )

            number_of_query_labels = int(
                query_labels.sum()
            )

            if number_of_query_labels == 0:
                continue

            neighbor_rows = (
                neighbor_target_rows[
                    query_index,
                    :k,
                ]
            )

            neighbor_labels = (
                targets[
                    neighbor_rows
                ].toarray()
                .astype(bool)
                .any(axis=0)
            )

            overlap = int(
                np.logical_and(
                    query_labels,
                    neighbor_labels,
                ).sum()
            )

            recalls.append(
                overlap
                / number_of_query_labels
            )

            hits.append(
                float(overlap > 0)
            )

        results[str(k)] = {
            "k": k,
            "evaluated_queries": (
                len(recalls)
            ),
            "hit_rate": float(
                np.mean(hits)
            ),
            "mean_recall": float(
                np.mean(recalls)
            ),
        }

    return results


def evaluate_configuration(
    protocol,
    pooling,
    embeddings,
    metadata,
):
    split_column = (
        f"{protocol}_split"
    )

    train_indices = np.flatnonzero(
        metadata[
            split_column
        ].eq("train").to_numpy()
    )

    valid_indices = np.flatnonzero(
        metadata[
            split_column
        ].eq("valid").to_numpy()
    )

    retriever = (
        ReactionEmbeddingRetriever(
            embeddings[
                train_indices
            ]
        )
    )

    local_neighbors, similarities = (
        retriever.search(
            embeddings[
                valid_indices
            ],
            top_k=max(TOP_K_VALUES),
        )
    )

    neighbor_indices = (
        train_indices[
            local_neighbors
        ]
    )

    query_types = metadata.iloc[
        valid_indices
    ][
        "reaction_type"
    ].to_numpy()

    neighbor_types = (
        metadata.iloc[
            neighbor_indices.reshape(-1)
        ][
            "reaction_type"
        ]
        .to_numpy()
        .reshape(
            neighbor_indices.shape
        )
    )

    query_centers = metadata.iloc[
        valid_indices
    ][
        "reaction_center_signature"
    ].to_numpy()

    neighbor_centers = (
        metadata.iloc[
            neighbor_indices.reshape(-1)
        ][
            "reaction_center_signature"
        ]
        .to_numpy()
        .reshape(
            neighbor_indices.shape
        )
    )

    train_centers = set(
        metadata.iloc[
            train_indices
        ][
            "reaction_center_signature"
        ]
    )

    type_results = {}

    for k in TOP_K_VALUES:
        hits = []

        for row_index in range(
            len(valid_indices)
        ):
            hits.append(
                query_types[row_index]
                in neighbor_types[
                    row_index,
                    :k,
                ]
            )

        type_results[str(k)] = {
            "hit_rate": float(
                np.mean(hits)
            )
        }

    type_mrr_at_10 = float(
        np.mean(
            [
                reciprocal_rank_at_k(
                    neighbor_types[
                        row_index
                    ]
                    == query_types[
                        row_index
                    ]
                )
                for row_index in range(
                    len(valid_indices)
                )
            ]
        )
    )

    center_eligible = np.asarray(
        [
            center in train_centers
            for center in (
                query_centers
            )
        ],
        dtype=bool,
    )

    center_results = {
        "eligible_queries": int(
            center_eligible.sum()
        )
    }

    for k in TOP_K_VALUES:
        hits = []

        for row_index in np.flatnonzero(
            center_eligible
        ):
            hits.append(
                query_centers[row_index]
                in neighbor_centers[
                    row_index,
                    :k,
                ]
            )

        center_results[str(k)] = {
            "hit_rate": (
                float(np.mean(hits))
                if hits
                else None
            )
        }

    condition_results = {}

    for target in TARGETS:
        condition_results[target] = (
            condition_retrieval_metrics(
                protocol=protocol,
                target=target,
                query_feature_indices=(
                    valid_indices
                ),
                neighbor_feature_indices=(
                    neighbor_indices
                ),
                metadata=metadata,
            )
        )

    neighbor_records = []

    for query_row, query_index in (
        enumerate(valid_indices)
    ):
        query = metadata.iloc[
            query_index
        ]

        for rank in range(
            neighbor_indices.shape[1]
        ):
            neighbor_index = (
                neighbor_indices[
                    query_row,
                    rank,
                ]
            )

            neighbor = metadata.iloc[
                neighbor_index
            ]

            neighbor_records.append(
                {
                    "protocol": protocol,
                    "pooling": pooling,
                    "query_feature_index": int(
                        query_index
                    ),
                    "query_transformation": (
                        query[
                            "transformation_signature"
                        ]
                    ),
                    "query_reaction_type": (
                        query["reaction_type"]
                    ),
                    "query_reaction_center": (
                        query[
                            "reaction_center_signature"
                        ]
                    ),
                    "rank": rank + 1,
                    "similarity": float(
                        similarities[
                            query_row,
                            rank,
                        ]
                    ),
                    "neighbor_feature_index": int(
                        neighbor_index
                    ),
                    "neighbor_transformation": (
                        neighbor[
                            "transformation_signature"
                        ]
                    ),
                    "neighbor_reaction_type": (
                        neighbor[
                            "reaction_type"
                        ]
                    ),
                    "neighbor_reaction_center": (
                        neighbor[
                            "reaction_center_signature"
                        ]
                    ),
                }
            )

    result = {
        "protocol": protocol,
        "pooling": pooling,
        "index_split": "train",
        "query_split": "valid",
        "test_used": False,
        "index_reactions": len(
            train_indices
        ),
        "query_reactions": len(
            valid_indices
        ),
        "nearest_similarity": {
            "minimum": float(
                similarities[:, 0].min()
            ),
            "mean": float(
                similarities[:, 0].mean()
            ),
            "median": float(
                np.median(
                    similarities[:, 0]
                )
            ),
            "maximum": float(
                similarities[:, 0].max()
            ),
        },
        "reaction_type": {
            "mrr_at_10": type_mrr_at_10,
            "top_k": type_results,
        },
        "reaction_center": (
            center_results
        ),
        "condition_retrieval": (
            condition_results
        ),
    }

    return result, neighbor_records


def main():
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive = np.load(
        FEATURE_PATH
    )

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    results = []
    neighbor_records = []

    print(
        "Reaction retrieval validation"
    )
    print("-----------------------------")
    print("Index split: train")
    print("Query split: valid")
    print("Test used: False")

    for protocol in PROTOCOLS:
        for pooling in (
            POOLING_METHODS
        ):
            result, neighbors = (
                evaluate_configuration(
                    protocol,
                    pooling,
                    archive[pooling],
                    metadata,
                )
            )

            results.append(result)
            neighbor_records.extend(
                neighbors
            )

            condition = result[
                "condition_retrieval"
            ]

            print(
                "\n"
                f"{protocol} | {pooling}"
            )

            print(
                "  reaction-type Hit@5:",
                round(
                    result[
                        "reaction_type"
                    ][
                        "top_k"
                    ]["5"]["hit_rate"],
                    4,
                ),
            )

            print(
                "  reaction-type MRR@10:",
                round(
                    result[
                        "reaction_type"
                    ]["mrr_at_10"],
                    4,
                ),
            )

            print(
                "  solvent recall@5:",
                round(
                    condition[
                        "solvent"
                    ]["5"]["mean_recall"],
                    4,
                ),
            )

            print(
                "  catalyst recall@5:",
                round(
                    condition[
                        "catalyst"
                    ]["5"]["mean_recall"],
                    4,
                ),
            )

            print(
                "  nearest similarity mean:",
                round(
                    result[
                        "nearest_similarity"
                    ]["mean"],
                    4,
                ),
            )

    pooling_scores = {}

    for pooling in POOLING_METHODS:
        values = []

        for result in results:
            if result[
                "pooling"
            ] != pooling:
                continue

            for target in TARGETS:
                values.append(
                    result[
                        "condition_retrieval"
                    ][target][
                        "5"
                    ]["mean_recall"]
                )

        pooling_scores[pooling] = float(
            np.mean(values)
        )

    selected_pooling = max(
        pooling_scores,
        key=pooling_scores.get,
    )

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "stage": "validation",
        "index_split": "train",
        "query_split": "valid",
        "test_used": False,
        "selection_metric": (
            "mean solvent/catalyst "
            "neighbor-label recall@5 "
            "across both split protocols"
        ),
        "pooling_scores": (
            pooling_scores
        ),
        "selected_pooling": (
            selected_pooling
        ),
        "results": results,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    pd.DataFrame(
        neighbor_records
    ).to_csv(
        NEIGHBOR_PATH,
        index=False,
    )

    print("\nPooling selection")
    print("-----------------")

    for pooling, score in (
        pooling_scores.items()
    ):
        print(
            f"{pooling:12s}",
            f"{score:.4f}",
        )

    print(
        "Selected pooling:",
        selected_pooling,
    )

    print("\nSaved:", REPORT_PATH)
    print("Saved:", NEIGHBOR_PATH)

    assert len(results) == 4

    assert all(
        result["test_used"] is False
        for result in results
    )

    print(
        "Retrieval validation passed."
    )


if __name__ == "__main__":
    main()