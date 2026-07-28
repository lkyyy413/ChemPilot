"""Audit complete solvent and catalyst label frequencies."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

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

OUTPUT_DIRECTORY = Path(
    "reports/day4/label_frequency"
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


def extract_identifier(message) -> tuple[str, str] | None:
    values = {}

    for identifier in message.identifiers:
        identifier_type = enum_name(
            identifier,
            "type",
        )

        value = identifier.value.strip()

        if value:
            values.setdefault(
                identifier_type,
                value,
            )

    for identifier_type in (
        "SMILES",
        "INCHI",
        "NAME",
    ):
        if identifier_type in values:
            return (
                identifier_type,
                values[identifier_type],
            )

    return None


def audit_dataset(
    dataset_name: str,
    path: Path,
) -> None:
    parquet = pq.ParquetFile(path)

    reaction_label_counts = {
        "SOLVENT": Counter(),
        "CATALYST": Counter(),
    }

    label_cardinality = {
        "SOLVENT": Counter(),
        "CATALYST": Counter(),
    }

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

            labels = {
                "SOLVENT": set(),
                "CATALYST": set(),
            }

            for reaction_input in (
                reaction.inputs.values()
            ):
                for component in (
                    reaction_input.components
                ):
                    role = enum_name(
                        component,
                        "reaction_role",
                    )

                    if role not in labels:
                        continue

                    identifier = extract_identifier(
                        component
                    )

                    if identifier is not None:
                        labels[role].add(
                            identifier
                        )

            for role in labels:
                label_cardinality[role][
                    len(labels[role])
                ] += 1

                reaction_label_counts[
                    role
                ].update(labels[role])

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for role, counts in (
        reaction_label_counts.items()
    ):
        output_path = (
            OUTPUT_DIRECTORY
            / f"{dataset_name}_"
              f"{role.lower()}_frequency.csv"
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(
                file,
                lineterminator="\n",
            )

            writer.writerow(
                [
                    "rank",
                    "identifier_type",
                    "label",
                    "reaction_count",
                ]
            )

            for rank, (
                (
                    identifier_type,
                    label,
                ),
                count,
            ) in enumerate(
                counts.most_common(),
                start=1,
            ):
                writer.writerow(
                    [
                        rank,
                        identifier_type,
                        label,
                        count,
                    ]
                )

        frequencies = list(counts.values())

        print(
            f"\n{dataset_name} | {role}"
        )
        print("Unique labels:", len(counts))
        print(
            "Reaction cardinality:",
            dict(
                sorted(
                    label_cardinality[
                        role
                    ].items()
                )
            ),
        )

        for threshold in (
            1,
            5,
            10,
            20,
            50,
            100,
        ):
            retained_classes = sum(
                frequency >= threshold
                for frequency in frequencies
            )

            retained_assignments = sum(
                frequency
                for frequency in frequencies
                if frequency >= threshold
            )

            print(
                f"Minimum frequency {threshold:3d}:",
                f"classes={retained_classes:4d}",
                f"assignments={retained_assignments:7d}",
            )

        print("Saved:", output_path)


def main() -> None:
    for dataset_name, path in (
        DATASETS.items()
    ):
        audit_dataset(
            dataset_name,
            path,
        )


if __name__ == "__main__":
    main()