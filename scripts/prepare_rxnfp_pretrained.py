"""Extract and audit the pinned RXNFP Reaction BERT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


EXPECTED_WHEEL_NAME = (
    "rxnfp-0.1.0-py3-none-any.whl"
)

EXPECTED_WHEEL_SIZE = 74_671_752

EXPECTED_WHEEL_SHA256 = (
    "c5c1e818add6f34539a6b29bc680c47c"
    "9e7311e9383d1b34ce901481e34b58cf"
)

PACKAGE_PREFIX = (
    "rxnfp/models/transformers/"
    "bert_pretrained/"
)

MEMBERS = {
    "config.json": (
        PACKAGE_PREFIX
        + "config.json"
    ),
    "pytorch_model.bin": (
        PACKAGE_PREFIX
        + "pytorch_model.bin"
    ),
    "vocab.txt": (
        PACKAGE_PREFIX
        + "vocab.txt"
    ),
    "LICENSE": (
        "rxnfp-0.1.0.dist-info/"
        "LICENSE"
    ),
}

DEFAULT_OUTPUT_DIRECTORY = Path(
    "artifacts/pretrained/day5/"
    "rxnfp_bert_pretrained"
)

DEFAULT_REPORT_PATH = Path(
    "reports/day5/"
    "rxnfp_pretrained_manifest.json"
)


def sha256_bytes(
    content: bytes,
) -> str:
    return hashlib.sha256(
        content
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wheel",
        type=Path,
        required=True,
        help=(
            "Path to the pinned "
            "rxnfp 0.1.0 wheel."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIRECTORY
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    wheel_path = (
        arguments.wheel.resolve()
    )

    if not wheel_path.exists():
        raise FileNotFoundError(
            wheel_path
        )

    if (
        wheel_path.name
        != EXPECTED_WHEEL_NAME
    ):
        raise ValueError(
            "Unexpected wheel name: "
            f"{wheel_path.name}"
        )

    wheel_size = (
        wheel_path.stat().st_size
    )

    wheel_sha256 = sha256_file(
        wheel_path
    )

    if wheel_size != EXPECTED_WHEEL_SIZE:
        raise ValueError(
            "Wheel size mismatch: "
            f"{wheel_size}"
        )

    if (
        wheel_sha256
        != EXPECTED_WHEEL_SHA256
    ):
        raise ValueError(
            "Wheel SHA256 mismatch: "
            f"{wheel_sha256}"
        )

    output_directory = (
        arguments.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_files = {}

    with ZipFile(wheel_path) as archive:
        archive_names = set(
            archive.namelist()
        )

        missing_members = [
            archive_name
            for archive_name
            in MEMBERS.values()
            if archive_name
            not in archive_names
        ]

        if missing_members:
            raise ValueError(
                "Missing wheel members: "
                f"{missing_members}"
            )

        for (
            output_name,
            archive_name,
        ) in MEMBERS.items():
            content = archive.read(
                archive_name
            )

            output_path = (
                output_directory
                / output_name
            )

            if output_path.exists():
                existing_sha256 = (
                    sha256_file(
                        output_path
                    )
                )

                expected_sha256 = (
                    sha256_bytes(
                        content
                    )
                )

                if (
                    existing_sha256
                    != expected_sha256
                ):
                    raise FileExistsError(
                        "Existing output has "
                        "different content: "
                        f"{output_path}"
                    )

            else:
                output_path.write_bytes(
                    content
                )

            extracted_files[
                output_name
            ] = {
                "path": str(
                    output_path
                ),
                "size_bytes": (
                    len(content)
                ),
                "sha256": (
                    sha256_bytes(
                        content
                    )
                ),
                "wheel_member": (
                    archive_name
                ),
            }

    config_path = (
        output_directory
        / "config.json"
    )

    with config_path.open(
        encoding="utf-8"
    ) as file:
        model_config = json.load(
            file
        )

    vocab_path = (
        output_directory
        / "vocab.txt"
    )

    vocabulary = (
        vocab_path.read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    report = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source": {
            "package": "rxnfp",
            "version": "0.1.0",
            "project_repository": (
                "https://github.com/"
                "rxn4chemistry/rxnfp"
            ),
            "distribution": (
                EXPECTED_WHEEL_NAME
            ),
            "wheel_size_bytes": (
                wheel_size
            ),
            "wheel_sha256": (
                wheel_sha256
            ),
            "license": "MIT",
            "license_note": (
                "The checkpoint is bundled "
                "inside the MIT-licensed "
                "rxnfp wheel; no separate "
                "model license file was "
                "present."
            ),
        },
        "checkpoint": {
            "name": (
                "bert_pretrained"
            ),
            "selection_reason": (
                "Reaction-SMILES masked-"
                "language-model pretraining "
                "without the source reaction-"
                "classification fine-tuning."
            ),
            "output_directory": str(
                output_directory
            ),
            "files": extracted_files,
        },
        "model_config": (
            model_config
        ),
        "vocabulary": {
            "size": len(
                vocabulary
            ),
            "first_tokens": (
                vocabulary[:20]
            ),
            "special_tokens_present": {
                token: (
                    token in vocabulary
                )
                for token in [
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]",
                    "[MASK]",
                ]
            },
        },
    }

    arguments.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with arguments.report.open(
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

    print("RXNFP checkpoint prepared")
    print("-------------------------")
    print("Wheel:", wheel_path)
    print("Wheel SHA256:", wheel_sha256)
    print(
        "Output:",
        output_directory,
    )

    for (
        name,
        metadata,
    ) in extracted_files.items():
        print(
            f"{name:20s}",
            f"{metadata['size_bytes']:>12,d}",
            metadata["sha256"],
        )

    print(
        "Vocabulary size:",
        len(vocabulary),
    )

    print(
        "Hidden size:",
        model_config.get(
            "hidden_size"
        ),
    )

    print(
        "Layers:",
        model_config.get(
            "num_hidden_layers"
        ),
    )

    print(
        "Attention heads:",
        model_config.get(
            "num_attention_heads"
        ),
    )

    print(
        "Maximum positions:",
        model_config.get(
            "max_position_embeddings"
        ),
    )

    print(
        "Saved manifest:",
        arguments.report,
    )


if __name__ == "__main__":
    main()
