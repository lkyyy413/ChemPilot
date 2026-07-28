from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import sparse

from train_reaction_condition_classifiers import (
    FEATURE_PATH,
    PROTOCOLS,
    TARGETS,
    train_one,
)


CANDIDATE_VALUES = (
    0.01,
    0.1,
    1.0,
    10.0,
)

OUTPUT_PATH = Path(
    "reports/day4/classification/"
    "logistic_c_search.csv"
)


def main() -> None:
    reaction_features = (
        sparse.load_npz(
            FEATURE_PATH
        ).tocsr()
    )

    records = []

    for protocol in PROTOCOLS:
        for target in TARGETS:
            for regularization_c in (
                CANDIDATE_VALUES
            ):
                print(
                    f"\n{protocol} | "
                    f"{target} | "
                    f"C={regularization_c}"
                )

                result = train_one(
                    protocol=protocol,
                    target=target,
                    evaluation_split="valid",
                    regularization_c=(
                        regularization_c
                    ),
                    max_iterations=2000,
                    n_jobs=8,
                    reaction_features=(
                        reaction_features
                    ),
                )

                model_metrics = result[
                    "model_metrics"
                ]

                baseline_metrics = result[
                    "frequency_baseline_metrics"
                ]

                record = {
                    "protocol": protocol,
                    "target": target,
                    "c": regularization_c,
                    "classes": result[
                        "classes"
                    ],
                    "train_samples": result[
                        "train_samples"
                    ],
                    "valid_samples": result[
                        "evaluation_samples"
                    ],
                    "micro_average_precision": (
                        model_metrics[
                            "micro_average_precision"
                        ]
                    ),
                    "macro_average_precision_observed": (
                        model_metrics[
                            "macro_average_precision_observed"
                        ]
                    ),
                    "mean_reciprocal_rank": (
                        model_metrics[
                            "mean_reciprocal_rank"
                        ]
                    ),
                    "hit_rate_at_1": (
                        model_metrics[
                            "top_k"
                        ]["1"]["hit_rate"]
                    ),
                    "hit_rate_at_3": (
                        model_metrics[
                            "top_k"
                        ]["3"]["hit_rate"]
                    ),
                    "hit_rate_at_5": (
                        model_metrics[
                            "top_k"
                        ]["5"]["hit_rate"]
                    ),
                    "recall_at_5": (
                        model_metrics[
                            "top_k"
                        ]["5"]["recall"]
                    ),
                    "frequency_hit_rate_at_5": (
                        baseline_metrics[
                            "top_k"
                        ]["5"]["hit_rate"]
                    ),
                    "training_seconds": (
                        result[
                            "training_seconds"
                        ]
                    ),
                    "maximum_iterations_used": (
                        result[
                            "maximum_iterations_used"
                        ]
                    ),
                    "converged": result[
                        "converged"
                    ],
                }

                records.append(record)

                print(
                    "  micro AP:",
                    round(
                        record[
                            "micro_average_precision"
                        ],
                        4,
                    ),
                )

                print(
                    "  HitRate@5:",
                    round(
                        record[
                            "hit_rate_at_5"
                        ],
                        4,
                    ),
                )

    results = pd.DataFrame(
        records
    )

    results["selected"] = False

    for (
        protocol,
        target,
    ), group in results.groupby(
        [
            "protocol",
            "target",
        ]
    ):
        # Micro AP 最大；并列时 HitRate@5 最大；
        # 再并列时选择更小的 C。
        selected_index = (
            group.sort_values(
                [
                    "micro_average_precision",
                    "hit_rate_at_5",
                    "c",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .index[0]
        )

        results.loc[
            selected_index,
            "selected",
        ] = True

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSelected configurations")
    print("-----------------------")

    selected = results.loc[
        results["selected"]
    ]

    print(
        selected[
            [
                "protocol",
                "target",
                "c",
                "micro_average_precision",
                "mean_reciprocal_rank",
                "hit_rate_at_5",
                "recall_at_5",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()