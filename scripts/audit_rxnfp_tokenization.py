"""Audit RXNFP tokenization on Day 5 reaction sequences."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chempilot.reactions.tokenization import (
    ReactionSmilesTokenizer,
)


DATA_PATH = Path(
    "data/processed/reactions/"
    "d9297630_condition_pairs.parquet"
)

SAMPLE_PATHS = {
    "transformation": Path(
        "data/processed/reactions/targets/"
        "transformation/solvent/"
        "samples.parquet"
    ),
    "reaction_center": Path(
        "data/processed/reactions/targets/"
        "reaction_center/solvent/"
        "samples.parquet"
    ),
}

CHECKPOINT_DIRECTORY = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)

REPORT_DIRECTORY = Path(
    "reports/day5/tokenization"
)

CSV_PATH = (
    REPORT_DIRECTORY
    / "rxnfp_tokenization_audit.csv"
)

JSON_PATH = (
    REPORT_DIRECTORY
    / "rxnfp_tokenization_audit.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_smiles(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [
        str(item)
        for item in value
        if item is not None
        and str(item).strip()
    ]


def canonical_sequence(row) -> str:
    reactants = sorted(
        normalize_smiles(
            row["reactant_smiles"]
        )
    )

    products = sorted(
        normalize_smiles(
            row["product_smiles"]
        )
    )

    return (
        ".".join(reactants)
        + ">>"
        + ".".join(products)
    )


def summarize_lengths(
    values: list[int],
) -> dict:
    array = np.asarray(
        values,
        dtype=np.int64,
    )

    result = {
        "count": int(array.size),
        "minimum": int(array.min()),
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
        "maximum": int(array.max()),
    }

    result["threshold_exceedance"] = {}

    # 两个 special tokens：
    # [CLS] reaction_tokens [SEP]
    for maximum_length in [
        128,
        256,
        512,
    ]:
        count = int(
            (
                array + 2
                > maximum_length
            ).sum()
        )

        result[
            "threshold_exceedance"
        ][str(maximum_length)] = {
            "count": count,
            "rate": (
                count
                / int(array.size)
            ),
        }

    return result


def main() -> None:
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_parquet(
        DATA_PATH
    )

    representative = (
        dataframe.sort_values(
            [
                "transformation_signature",
                "representative_reaction_id",
            ]
        )
        .drop_duplicates(
            "transformation_signature"
        )
        .copy()
    )

    representative[
        "canonical_reaction"
    ] = representative.apply(
        canonical_sequence,
        axis=1,
    )

    protocol_samples = {}

    for protocol, path in (
        SAMPLE_PATHS.items()
    ):
        samples = pd.read_parquet(path)

        protocol_samples[protocol] = {
            row[
                "transformation_signature"
            ]: row["split"]
            for _, row in samples.iterrows()
        }

    transformation_set = set(
        protocol_samples[
            "transformation"
        ]
    )

    reaction_center_set = set(
        protocol_samples[
            "reaction_center"
        ]
    )

    assert (
        transformation_set
        == reaction_center_set
    )

    modeling_transformations = (
        transformation_set
    )

    modeling = representative.loc[
        representative[
            "transformation_signature"
        ].isin(
            modeling_transformations
        )
    ].copy()

    tokenizer = ReactionSmilesTokenizer(
        CHECKPOINT_DIRECTORY
        / "vocab.txt"
    )

    records = []
    unknown_token_counter = Counter()

    for _, row in modeling.iterrows():
        reaction = row[
            "canonical_reaction"
        ]

        raw_tokens = (
            tokenizer.smiles_tokenizer
            .tokenize_checked(
                reaction
            )
        )

        vocabulary_tokens = (
            tokenizer.convert_tokens_to_ids(
                raw_tokens
            )
        )

        unknown_positions = [
            index
            for index, token_id in enumerate(
                vocabulary_tokens
            )
            if token_id
            == tokenizer.unk_token_id
        ]

        unknown_tokens = [
            raw_tokens[index]
            for index in unknown_positions
        ]

        unknown_token_counter.update(
            unknown_tokens
        )

        encoded = tokenizer(
            reaction,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )

        encoded_ids = encoded[
            "input_ids"
        ]

        records.append(
            {
                "transformation_signature": (
                    row[
                        "transformation_signature"
                    ]
                ),
                "reaction_center_signature": (
                    row[
                        "reaction_center_signature"
                    ]
                ),
                "reaction_type": (
                    row["reaction_type"]
                ),
                "canonical_reaction": reaction,
                "raw_token_count": len(
                    raw_tokens
                ),
                "encoded_token_count": len(
                    encoded_ids
                ),
                "unknown_token_count": len(
                    unknown_tokens
                ),
                "unknown_tokens": (
                    unknown_tokens
                ),
                "transformation_split": (
                    protocol_samples[
                        "transformation"
                    ][
                        row[
                            "transformation_signature"
                        ]
                    ]
                ),
                "reaction_center_split": (
                    protocol_samples[
                        "reaction_center"
                    ][
                        row[
                            "transformation_signature"
                        ]
                    ]
                ),
            }
        )

    audit = pd.DataFrame(records)

    total_raw_tokens = int(
        audit["raw_token_count"].sum()
    )

    total_unknown_tokens = int(
        audit[
            "unknown_token_count"
        ].sum()
    )

    sequences_with_unknown = int(
        audit[
            "unknown_token_count"
        ].gt(0).sum()
    )

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input": {
            "path": str(DATA_PATH),
            "sha256": sha256_file(
                DATA_PATH
            ),
            "modeling_transformations": (
                len(audit)
            ),
        },
        "checkpoint": {
            "directory": str(
                CHECKPOINT_DIRECTORY
            ),
            "vocabulary_path": str(
                CHECKPOINT_DIRECTORY
                / "vocab.txt"
            ),
            "vocabulary_sha256": (
                sha256_file(
                    CHECKPOINT_DIRECTORY
                    / "vocab.txt"
                )
            ),
            "vocabulary_size": (
                tokenizer.vocab_size
            ),
        },
        "tokenization": {
            "pattern": (
                tokenizer
                .smiles_tokenizer
                .pattern
            ),
            "special_tokens_per_sequence": 2,
            "total_raw_tokens": (
                total_raw_tokens
            ),
            "total_unknown_tokens": (
                total_unknown_tokens
            ),
            "unknown_token_rate": (
                total_unknown_tokens
                / total_raw_tokens
                if total_raw_tokens
                else 0.0
            ),
            "sequences_with_unknown": (
                sequences_with_unknown
            ),
            "sequence_unknown_rate": (
                sequences_with_unknown
                / len(audit)
            ),
            "unknown_token_frequencies": [
                {
                    "token": token,
                    "count": count,
                }
                for token, count in (
                    unknown_token_counter
                    .most_common()
                )
            ],
            "raw_token_lengths": (
                summarize_lengths(
                    audit[
                        "raw_token_count"
                    ].tolist()
                )
            ),
            "encoded_token_lengths": {
                "minimum": int(
                    audit[
                        "encoded_token_count"
                    ].min()
                ),
                "maximum": int(
                    audit[
                        "encoded_token_count"
                    ].max()
                ),
            },
        },
        "recommended_configuration": {
            "max_length": 256,
            "reason": (
                "Selected only if the "
                "256-token truncation rate "
                "is zero."
            ),
        },
    }

    audit.to_csv(
        CSV_PATH,
        index=False,
    )

    with JSON_PATH.open(
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

    lengths = report[
        "tokenization"
    ]["raw_token_lengths"]

    print(
        "RXNFP tokenizer audit"
    )
    print("---------------------")
    print(
        "Modeling transformations:",
        len(audit),
    )
    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )
    print(
        "Total raw tokens:",
        total_raw_tokens,
    )
    print(
        "Unknown tokens:",
        total_unknown_tokens,
    )
    print(
        "Sequences with unknown:",
        sequences_with_unknown,
    )

    print("\nRaw token lengths:")

    for name in [
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
            f"  {name:8s}",
            lengths[name],
        )

    print("\nTruncation audit:")

    for threshold, values in (
        lengths[
            "threshold_exceedance"
        ].items()
    ):
        print(
            f"  max_length={threshold:>3s}",
            f"{values['count']:3d}",
            f"{values['rate']:.2%}",
        )

    print("\nMost common unknown tokens:")

    for token, count in (
        unknown_token_counter
        .most_common(20)
    ):
        print(
            f"  {count:5d}",
            repr(token),
        )

    assert len(audit) == 381
    assert total_unknown_tokens == 0

    assert (
        lengths[
            "threshold_exceedance"
        ]["256"]["count"]
        == 0
    )

    print(
        "\nRXNFP tokenizer audit passed."
    )
    print("Saved:", CSV_PATH)
    print("Saved:", JSON_PATH)


if __name__ == "__main__":
    main()