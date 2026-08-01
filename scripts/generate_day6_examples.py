"""Generate reproducible Day 6 inference examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

from chempilot.service.prediction import (
    PredictionService,
)
from chempilot.service.registry import (
    DEFAULT_INFERENCE_CONFIG_PATH,
    ModelRegistry,
)
from chempilot.service.schemas import (
    PredictionRequest,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "day6"
    / "examples"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate molecule and reaction "
            "examples using the unified "
            "ChemPilot service."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            DEFAULT_INFERENCE_CONFIG_PATH
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIRECTORY
        ),
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(
            1024 * 1024
        ):
            digest.update(block)

    return digest.hexdigest()


def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()

    registry = ModelRegistry(
        config_path=arguments.config
    )
    service = PredictionService(
        registry
    )

    molecule_request = (
        PredictionRequest(
            molecule_smiles=(
                "CC(=O)Oc1ccccc1C(=O)O"
            )
        )
    )

    reaction_request = (
        PredictionRequest(
            reactant_smiles=[
                "Brc1ccc2ncccc2c1",
                "O=S([O-])C1CC1.[Na+]",
            ],
            product_smiles=[
                "c1cnc2ccc(C3CC3)cc2c1",
            ],
            reaction_protocol=(
                "reaction_center"
            ),
            top_k=5,
        )
    )

    print(
        "Generating molecule example..."
    )

    molecule_result = service.predict(
        molecule_request,
        request_id=(
            "day6-example-molecule"
        ),
    )

    print(
        "Generating reaction example..."
    )

    reaction_result = service.predict(
        reaction_request,
        request_id=(
            "day6-example-reaction"
        ),
    )

    output_directory = (
        arguments.output_directory
    )

    molecule_path = (
        output_directory
        / "molecule_prediction.json"
    )

    reaction_path = (
        output_directory
        / "reaction_prediction.json"
    )

    save_json(
        molecule_path,
        molecule_result.model_dump(
            mode="json"
        ),
    )

    save_json(
        reaction_path,
        reaction_result.model_dump(
            mode="json"
        ),
    )

    manifest_path = (
        output_directory
        / "example_manifest.json"
    )

    manifest = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "configuration": {
            "path": str(
                arguments.config
            ),
            "sha256": sha256_file(
                arguments.config
            ),
        },
        "examples": {
            "molecule": {
                "request": (
                    molecule_request
                    .model_dump(
                        mode="json"
                    )
                ),
                "response_path": str(
                    molecule_path
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),
                "response_sha256": (
                    sha256_file(
                        molecule_path
                    )
                ),
            },
            "reaction": {
                "request": (
                    reaction_request
                    .model_dump(
                        mode="json"
                    )
                ),
                "response_path": str(
                    reaction_path
                    .relative_to(
                        PROJECT_ROOT
                    )
                ),
                "response_sha256": (
                    sha256_file(
                        reaction_path
                    )
                ),
            },
        },
        "interpretation": {
            "solubility": (
                "Predicted LogS is reported in "
                "log10(mol/L)."
            ),
            "condition_scores": (
                "Condition scores are ranking "
                "scores, not calibrated "
                "probabilities."
            ),
            "retrieval_similarity": (
                "RXNFP cosine similarity is not "
                "a probability of reaction "
                "success."
            ),
            "historical_response": (
                "ORD LC area percent at 280 nm "
                "is not isolated reaction yield."
            ),
            "synthesizability": (
                "The reported synthesis risk is "
                "a lightweight assessment, not "
                "multi-step retrosynthesis."
            ),
        },
    }

    save_json(
        manifest_path,
        manifest,
    )

    print("\nSaved:")
    print(molecule_path)
    print(reaction_path)
    print(manifest_path)


if __name__ == "__main__":
    main()