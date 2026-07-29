"""Audit canonical reaction sequences for Day 5 Transformer models."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


DATASET_PATH = Path(
    "data/processed/reactions/"
    "d9297630_condition_pairs.parquet"
)

TARGET_ROOT = Path(
    "data/processed/reactions/targets"
)

OUTPUT_DIRECTORY = Path(
    "reports/day5/reaction_sequences"
)

SEQUENCE_TABLE_PATH = (
    OUTPUT_DIRECTORY
    / "reaction_sequence_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "reaction_sequence_audit.json"
)

# Common reaction-SMILES lexical tokenizer.
# This is only a length proxy. The actual pretrained tokenizer
# will be audited after the model family is selected.
TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]"
    r"|Br?|Cl?"
    r"|N|O|S|P|F|I"
    r"|b|c|n|o|s|p"
    r"|\(|\)|\."
    r"|=|#|-|\+"
    r"|\\|/"
    r"|:|~|@|\?"
    r"|>"
    r"|\*|\$"
    r"|%[0-9]{2}"
    r"|[0-9])"
)

LENGTH_THRESHOLDS = [
    128,
    256,
    512,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_smiles(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    elif isinstance(value, np.ndarray):
        value = value.tolist()

    elif not isinstance(
        value,
        (list, tuple),
    ):
        value = [value]

    return sorted(
        {
            str(item).strip()
            for item in value
            if item is not None
            and str(item).strip()
        }
    )


def build_reaction_smiles(
    reactants: Any,
    products: Any,
) -> str | None:
    reactant_list = normalize_smiles(
        reactants
    )

    product_list = normalize_smiles(
        products
    )

    if (
        not reactant_list
        or not product_list
    ):
        return None

    return (
        ".".join(reactant_list)
        + ">>"
        + ".".join(product_list)
    )


def lexical_tokens(
    sequence: str,
) -> list[str]:
    return TOKEN_PATTERN.findall(
        sequence
    )


def unmatched_character_count(
    sequence: str,
    tokens: list[str],
) -> int:
    return (
        len(sequence)
        - sum(
            len(token)
            for token in tokens
        )
    )


def molecule_is_valid(
    smiles: str,
) -> bool:
    molecule = Chem.MolFromSmiles(
        smiles
    )

    return molecule is not None


def sequence_is_valid(
    sequence: str,
) -> bool:
    if sequence.count(">>") != 1:
        return False

    reactants, products = (
        sequence.split(">>")
    )

    reactant_components = [
        component
        for component in (
            reactants.split(".")
        )
        if component
    ]

    product_components = [
        component
        for component in (
            products.split(".")
        )
        if component
    ]

    if (
        not reactant_components
        or not product_components
    ):
        return False

    return all(
        molecule_is_valid(component)
        for component in (
            reactant_components
            + product_components
        )
    )


def describe_lengths(
    values: pd.Series,
) -> dict[str, Any]:
    array = values.to_numpy(
        dtype=float
    )

    result: dict[str, Any] = {
        "count": int(len(array)),
        "minimum": int(
            np.min(array)
        ),
        "p25": float(
            np.percentile(array, 25)
        ),
        "median": float(
            np.percentile(array, 50)
        ),
        "p75": float(
            np.percentile(array, 75)
        ),
        "p90": float(
            np.percentile(array, 90)
        ),
        "p95": float(
            np.percentile(array, 95)
        ),
        "p99": float(
            np.percentile(array, 99)
        ),
        "maximum": int(
            np.max(array)
        ),
    }

    result[
        "threshold_exceedance"
    ] = {
        str(threshold): {
            "count": int(
                np.sum(
                    array > threshold
                )
            ),
            "rate": float(
                np.mean(
                    array > threshold
                )
            ),
        }
        for threshold in (
            LENGTH_THRESHOLDS
        )
    }

    return result


def load_split_assignments(
    protocol: str,
) -> pd.DataFrame:
    path = (
        TARGET_ROOT
        / protocol
        / "solvent"
        / "samples.parquet"
    )

    samples = pd.read_parquet(
        path,
        columns=[
            "transformation_signature",
            "split",
        ],
    )

    return samples.rename(
        columns={
            "split": (
                f"{protocol}_split"
            )
        }
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_parquet(
        DATASET_PATH
    )

    dataframe = dataframe.copy()

    dataframe[
        "canonical_reaction_smiles"
    ] = [
        build_reaction_smiles(
            reactants,
            products,
        )
        for reactants, products in zip(
            dataframe[
                "reactant_smiles"
            ],
            dataframe[
                "product_smiles"
            ],
        )
    ]

    grouped = dataframe.groupby(
        "transformation_signature",
        sort=False,
    )

    records: list[dict[str, Any]] = []

    for (
        transformation_signature,
        group,
    ) in grouped:
        canonical_sequences = (
            group[
                "canonical_reaction_smiles"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if len(canonical_sequences) != 1:
            raise ValueError(
                "Expected exactly one "
                "canonical sequence for "
                f"{transformation_signature}; "
                f"received "
                f"{len(canonical_sequences)}."
            )

        sequence = canonical_sequences[0]

        tokens = lexical_tokens(
            sequence
        )

        mapped_sequences = (
            group[
                "reaction_smiles_mapped"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        mapped_character_lengths = [
            len(value)
            for value in mapped_sequences
        ]

        mapped_token_lengths = [
            len(
                lexical_tokens(value)
            )
            for value in mapped_sequences
        ]

        records.append(
            {
                "transformation_signature": (
                    transformation_signature
                ),
                "reaction_type": (
                    group[
                        "reaction_type"
                    ].iloc[0]
                ),
                "reaction_center_signature": (
                    group[
                        "reaction_center_signature"
                    ].iloc[0]
                ),
                "canonical_reaction_smiles": (
                    sequence
                ),
                "canonical_character_length": (
                    len(sequence)
                ),
                "canonical_token_length": (
                    len(tokens)
                ),
                "unmatched_character_count": (
                    unmatched_character_count(
                        sequence,
                        tokens,
                    )
                ),
                "rdkit_valid": (
                    sequence_is_valid(
                        sequence
                    )
                ),
                "condition_pair_count": (
                    int(len(group))
                ),
                "represented_experiment_count": (
                    int(
                        group[
                            "replicate_count"
                        ].sum()
                    )
                ),
                "mapped_sequence_variants": (
                    len(mapped_sequences)
                ),
                "mapped_character_length_max": (
                    max(
                        mapped_character_lengths,
                        default=0,
                    )
                ),
                "mapped_token_length_max": (
                    max(
                        mapped_token_lengths,
                        default=0,
                    )
                ),
            }
        )

    audit = pd.DataFrame(
        records
    )

    for protocol in [
        "transformation",
        "reaction_center",
    ]:
        audit = audit.merge(
            load_split_assignments(
                protocol
            ),
            on=(
                "transformation_signature"
            ),
            how="left",
            validate="one_to_one",
        )

    audit[
        "is_modeling_transformation"
    ] = audit[
        "transformation_split"
    ].notna()

    audit = audit.sort_values(
        [
            "is_modeling_transformation",
            "canonical_token_length",
            "transformation_signature",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    modeling = audit.loc[
        audit[
            "is_modeling_transformation"
        ]
    ].copy()

    report = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "input": {
            "path": str(
                DATASET_PATH
            ),
            "sha256": sha256_file(
                DATASET_PATH
            ),
        },
        "sequence_definition": {
            "format": (
                "sorted canonical reactants"
                ">>"
                "sorted canonical products"
            ),
            "includes_reagents": False,
            "includes_solvents": False,
            "includes_catalysts": False,
            "atom_mapping_used": False,
            "lexical_tokenizer": (
                TOKEN_PATTERN.pattern
            ),
            "lexical_tokenizer_note": (
                "Length proxy only; the "
                "selected pretrained tokenizer "
                "must be audited separately."
            ),
        },
        "all_transformations": {
            "rows": int(
                len(audit)
            ),
            "unique_sequences": int(
                audit[
                    "canonical_reaction_smiles"
                ].nunique()
            ),
            "rdkit_invalid": int(
                (~audit["rdkit_valid"]).sum()
            ),
            "lexically_unmatched": int(
                audit[
                    "unmatched_character_count"
                ].gt(0).sum()
            ),
            "mapped_sequence_conflicts": int(
                audit[
                    "mapped_sequence_variants"
                ].gt(1).sum()
            ),
            "character_lengths": (
                describe_lengths(
                    audit[
                        "canonical_character_length"
                    ]
                )
            ),
            "token_lengths": (
                describe_lengths(
                    audit[
                        "canonical_token_length"
                    ]
                )
            ),
        },
        "modeling_transformations": {
            "rows": int(
                len(modeling)
            ),
            "unique_sequences": int(
                modeling[
                    "canonical_reaction_smiles"
                ].nunique()
            ),
            "rdkit_invalid": int(
                (
                    ~modeling[
                        "rdkit_valid"
                    ]
                ).sum()
            ),
            "lexically_unmatched": int(
                modeling[
                    "unmatched_character_count"
                ].gt(0).sum()
            ),
            "mapped_sequence_conflicts": int(
                modeling[
                    "mapped_sequence_variants"
                ].gt(1).sum()
            ),
            "character_lengths": (
                describe_lengths(
                    modeling[
                        "canonical_character_length"
                    ]
                )
            ),
            "token_lengths": (
                describe_lengths(
                    modeling[
                        "canonical_token_length"
                    ]
                )
            ),
            "transformation_split_counts": {
                str(key): int(value)
                for key, value in (
                    modeling[
                        "transformation_split"
                    ]
                    .value_counts()
                    .items()
                )
            },
            "reaction_center_split_counts": {
                str(key): int(value)
                for key, value in (
                    modeling[
                        "reaction_center_split"
                    ]
                    .value_counts()
                    .items()
                )
            },
        },
    }

    audit.to_csv(
        SEQUENCE_TABLE_PATH,
        index=False,
    )

    with SUMMARY_PATH.open(
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

    print(
        "Reaction-sequence audit"
    )
    print(
        "-----------------------"
    )
    print(
        "All transformations:",
        f"{len(audit):,}",
    )
    print(
        "Modeling transformations:",
        f"{len(modeling):,}",
    )
    print(
        "RDKit-invalid modeling sequences:",
        (
            ~modeling[
                "rdkit_valid"
            ]
        ).sum(),
    )
    print(
        "Lexically unmatched modeling "
        "sequences:",
        modeling[
            "unmatched_character_count"
        ].gt(0).sum(),
    )
    print(
        "Mapped conflicts in modeling set:",
        modeling[
            "mapped_sequence_variants"
        ].gt(1).sum(),
    )

    token_statistics = report[
        "modeling_transformations"
    ]["token_lengths"]

    print("\nModeling lexical-token lengths:")

    for key in [
        "minimum",
        "p25",
        "median",
        "p75",
        "p90",
        "p95",
        "p99",
        "maximum",
    ]:
        print(
            f"  {key:8s}",
            token_statistics[key],
        )

    print("\nThreshold exceedance:")

    for threshold, values in (
        token_statistics[
            "threshold_exceedance"
        ].items()
    ):
        print(
            f"  >{threshold:3s}",
            f"{values['count']:4d}",
            f"{values['rate']:.2%}",
        )

    print(
        "\nSaved:",
        SEQUENCE_TABLE_PATH,
    )
    print(
        "Saved:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
