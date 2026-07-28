"""Audit ORD product measurement types and payload fields."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from google.protobuf.json_format import (
    MessageToDict,
)

from ord_schema.proto import reaction_pb2


DATASETS = {
    "805ad863": Path(
        "/tmp/ord-data-source/data/80/"
        "ord_dataset-805ad863feef48579d95d86a728035f4.parquet"
    ),
    "d9297630": Path(
        "/tmp/ord-data-source/data/d9/"
        "ord_dataset-d92976309c3a48a3a64a4cf5e7048086.parquet"
    ),
}

OUTPUT_PATH = Path(
    "reports/day4/"
    "ord_measurement_audit.json"
)


def enum_name(message, field_name: str) -> str:
    field = message.DESCRIPTOR.fields_by_name[
        field_name
    ]

    value = int(getattr(message, field_name))

    descriptor = (
        field.enum_type.values_by_number.get(
            value
        )
    )

    if descriptor is None:
        return f"UNKNOWN_{value}"

    return descriptor.name


def message_to_dict(message) -> dict:
    return MessageToDict(
        message,
        preserving_proto_field_name=True,
        use_integers_for_enums=False,
    )


def audit_dataset(path: Path) -> dict:
    parquet = pq.ParquetFile(path)

    reaction_counts = Counter()
    measurement_counts = Counter()
    analysis_key_counts = Counter()
    detail_counts = Counter()
    payload_field_counts = Counter()

    examples = defaultdict(list)

    for row_group_index in range(
        parquet.num_row_groups
    ):
        table = parquet.read_row_group(
            row_group_index,
            columns=["reaction"],
        )

        for serialized in (
            table["reaction"].to_pylist()
        ):
            reaction = reaction_pb2.Reaction()
            reaction.ParseFromString(
                serialized
            )

            reaction_counts["total"] += 1

            reaction_has_measurement = False
            reaction_has_desired_measurement = (
                False
            )

            for outcome in reaction.outcomes:
                for product in outcome.products:
                    scope = (
                        "desired"
                        if product.is_desired_product
                        else "other"
                    )

                    for measurement in (
                        product.measurements
                    ):
                        reaction_has_measurement = (
                            True
                        )

                        if scope == "desired":
                            reaction_has_desired_measurement = (
                                True
                            )

                        measurement_type = enum_name(
                            measurement,
                            "type",
                        )

                        key = (
                            scope,
                            measurement_type,
                        )

                        measurement_counts[key] += 1

                        if measurement.analysis_key:
                            analysis_key_counts[
                                (
                                    scope,
                                    measurement_type,
                                    measurement.analysis_key,
                                )
                            ] += 1

                        if measurement.details:
                            detail_counts[
                                (
                                    scope,
                                    measurement_type,
                                    measurement.details,
                                )
                            ] += 1

                        populated_fields = tuple(
                            sorted(
                                descriptor.name
                                for descriptor, _ in (
                                    measurement.ListFields()
                                )
                            )
                        )

                        payload_field_counts[
                            (
                                scope,
                                measurement_type,
                                populated_fields,
                            )
                        ] += 1

                        if len(examples[key]) < 3:
                            examples[key].append(
                                message_to_dict(
                                    measurement
                                )
                            )

            if reaction_has_measurement:
                reaction_counts[
                    "has_any_measurement"
                ] += 1

            if reaction_has_desired_measurement:
                reaction_counts[
                    "has_desired_measurement"
                ] += 1

    return {
        "number_of_reactions": (
            reaction_counts["total"]
        ),
        "reactions_with_any_measurement": (
            reaction_counts[
                "has_any_measurement"
            ]
        ),
        "reactions_with_desired_measurement": (
            reaction_counts[
                "has_desired_measurement"
            ]
        ),
        "measurement_counts": [
            {
                "scope": scope,
                "type": measurement_type,
                "count": count,
            }
            for (
                scope,
                measurement_type,
            ), count in (
                measurement_counts.most_common()
            )
        ],
        "payload_field_counts": [
            {
                "scope": scope,
                "type": measurement_type,
                "fields": list(fields),
                "count": count,
            }
            for (
                scope,
                measurement_type,
                fields,
            ), count in (
                payload_field_counts.most_common()
            )
        ],
        "top_analysis_keys": [
            {
                "scope": scope,
                "type": measurement_type,
                "analysis_key": analysis_key,
                "count": count,
            }
            for (
                scope,
                measurement_type,
                analysis_key,
            ), count in (
                analysis_key_counts.most_common(20)
            )
        ],
        "top_details": [
            {
                "scope": scope,
                "type": measurement_type,
                "details": details,
                "count": count,
            }
            for (
                scope,
                measurement_type,
                details,
            ), count in (
                detail_counts.most_common(20)
            )
        ],
        "examples": {
            f"{scope}|{measurement_type}": values
            for (
                scope,
                measurement_type,
            ), values in examples.items()
        },
    }


def main() -> None:
    report = {}

    for dataset_name, path in DATASETS.items():
        print(f"Auditing {dataset_name}...")

        result = audit_dataset(path)
        report[dataset_name] = result

        print(
            "  reactions:",
            result["number_of_reactions"],
        )

        print(
            "  desired measurements:",
            result[
                "reactions_with_desired_measurement"
            ],
        )

        print("  measurement types:")

        for item in result[
            "measurement_counts"
        ]:
            print(
                "   ",
                item["scope"],
                item["type"],
                item["count"],
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()