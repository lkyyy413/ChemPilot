"""Generate the Day 6 unified-inference report."""

from __future__ import annotations

import json
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

EXAMPLE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "day6"
    / "examples"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day6"
    / "day6_unified_inference_report.md"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def short_label(
    value: str,
    maximum_length: int = 80,
) -> str:
    if len(value) <= maximum_length:
        return value

    return (
        value[
            : maximum_length - 3
        ]
        + "..."
    )


def main() -> None:
    molecule_response = load_json(
        EXAMPLE_ROOT
        / "molecule_prediction.json"
    )

    reaction_response = load_json(
        EXAMPLE_ROOT
        / "reaction_prediction.json"
    )

    manifest = load_json(
        EXAMPLE_ROOT
        / "example_manifest.json"
    )

    molecule = molecule_response[
        "molecule"
    ]
    reaction = reaction_response[
        "reaction"
    ]

    descriptors = molecule[
        "descriptors"
    ]
    solubility = molecule[
        "solubility"
    ]
    drug_likeness = molecule[
        "drug_likeness"
    ]
    molecule_risk = molecule[
        "synthesizability"
    ]

    conditions = reaction[
        "conditions"
    ]
    retrieval = reaction[
        "retrieval"
    ]
    confidence = reaction[
        "confidence"
    ]
    reaction_risk = reaction[
        "synthesizability"
    ]

    top_solvent = conditions[
        "solvent"
    ][
        "top_k"
    ][0]

    top_catalyst = conditions[
        "catalyst"
    ][
        "top_k"
    ][0]

    nearest = retrieval[
        "neighbors"
    ][0]

    lines = [
        "# Day 6: Unified ChemPilot Inference Service",
        "",
        (
            "Generated at "
            f"{datetime.now(timezone.utc).isoformat()}."
        ),
        "",
        "## Objective",
        "",
        (
            "Day 6 combines the molecular-property and reaction models "
            "from Days 1–5 into one configuration-driven inference "
            "service."
        ),
        "",
        (
            "A molecule SMILES request returns aqueous-solubility "
            "prediction, molecular descriptors, Lipinski drug-likeness, "
            "SA score, rare-fragment count, structural complexity, and "
            "a lightweight synthesizability-risk assessment."
        ),
        "",
        (
            "When reactants and target products are supplied, the "
            "service additionally returns Top-K solvent and catalyst "
            "rankings, applicability-domain information, similar "
            "historical reactions, qualitative confidence, and a "
            "context-aware synthesis-risk assessment."
        ),
        "",
        (
            "The synthesizability output is not a multi-step "
            "retrosynthesis plan and is not a guarantee that a compound "
            "or reaction is experimentally feasible."
        ),
        "",
        "## Architecture",
        "",
        "```text",
        "PredictionRequest",
        "        |",
        "PredictionService",
        "        |",
        "        +-- MoleculeStandardizer",
        "        +-- MoleculeRiskAnalyzer",
        "        +-- SolubilityPredictor",
        "        +-- ReactionConditionPredictor",
        "        +-- SimilarReactionSearch",
        "        |",
        "ModelRegistry -- configs/inference.yaml",
        "        |",
        "UnifiedPredictionResponse",
        "```",
        "",
        "The main composition interfaces are:",
        "",
        "- `BaseFeaturizer`: validates feature names, dimensions, rows, and finite values.",
        "- `BasePredictor`: defines a reusable prediction interface.",
        "- `ModelRegistry`: resolves configuration, lazily loads models, caches instances, and reports artifact availability.",
        "- `PredictionService`: orchestrates molecule and reaction inference and maps outputs into strict Pydantic schemas.",
        "- `FastAPI`: exposes structured HTTP JSON endpoints and OpenAPI documentation.",
        "",
        "## Configuration",
        "",
        (
            "Model paths, supported protocols, device selection, "
            "drug-like scope, and synthesis-risk thresholds are stored "
            "in `configs/inference.yaml`."
        ),
        "",
        (
            "Configuration SHA256: `"
            f"{manifest['configuration']['sha256']}`"
        ),
        "",
        "The selected production components are:",
        "",
        "- Solubility: Day 2 scaffold-split XGBoost with descriptor-plus-ECFP features.",
        "- Condition recommendation: Day 4 Morgan logistic multi-label classifiers.",
        "- Historical retrieval: Day 5 RXNFP CLS embeddings.",
        "",
        "## Molecular inference example",
        "",
        (
            f"Input SMILES: `{molecule['input_smiles']}`"
        ),
        "",
        (
            f"Canonical SMILES: `{molecule['canonical_smiles']}`"
        ),
        "",
        "| Output | Value |",
        "|---|---:|",
        (
            "| Predicted LogS | "
            f"{solubility['predicted_log_s']:.4f} "
            "log10(mol/L) |"
        ),
        (
            "| Molecular weight | "
            f"{descriptors['molecular_weight']:.3f} |"
        ),
        (
            "| MolLogP | "
            f"{descriptors['log_p']:.4f} |"
        ),
        (
            "| TPSA | "
            f"{descriptors['tpsa']:.3f} |"
        ),
        (
            "| Lipinski pass | "
            f"{drug_likeness['lipinski_pass']} |"
        ),
        (
            "| SA score | "
            f"{molecule_risk['sa_score']:.4f} |"
        ),
        (
            "| Rare fragments | "
            f"{molecule_risk['rare_fragment_count']} |"
        ),
        (
            "| Molecular complexity | "
            f"{molecule_risk['molecular_complexity']:.4f} |"
        ),
        (
            "| Synthesis-risk level | "
            f"{molecule_risk['risk_level']} |"
        ),
        "",
        (
            "The LogS model uses 2,058 features in the exact Day 2 "
            "training order: 10 global RDKit descriptors followed by "
            "2,048 ECFP4/Morgan bits."
        ),
        "",
        "## Reaction inference example",
        "",
        "Canonical reactants:",
        "",
    ]

    for reactant in reaction[
        "canonical_reactants"
    ]:
        lines.append(
            f"- `{reactant}`"
        )

    lines.extend(
        [
            "",
            "Canonical products:",
            "",
        ]
    )

    for product in reaction[
        "canonical_products"
    ]:
        lines.append(
            f"- `{product}`"
        )

    lines.extend(
        [
            "",
            "| Output | Result |",
            "|---|---|",
            (
                "| Top solvent | "
                f"`{short_label(top_solvent['label'])}` "
                f"(score {top_solvent['ranking_score']:.4f}) |"
            ),
            (
                "| Top catalyst | "
                f"`{short_label(top_catalyst['label'])}` "
                f"(score {top_catalyst['ranking_score']:.4f}) |"
            ),
            (
                "| Nearest historical reaction type | "
                f"{nearest.get('reaction_type')} |"
            ),
            (
                "| Nearest RXNFP similarity | "
                f"{nearest['similarity']:.4f} |"
            ),
            (
                "| Condition uncertainty | "
                f"{reaction_risk['condition_uncertainty']:.4f} |"
            ),
            (
                "| Qualitative confidence | "
                f"{confidence['level']} |"
            ),
            (
                "| Context-aware synthesis risk | "
                f"{reaction_risk['risk_level']} |"
            ),
            "",
            "## Confidence and uncertainty",
            "",
            (
                "For each target, the ranking uncertainty is defined "
                "from the Top-1 versus Top-2 score margin:"
            ),
            "",
            "```text",
            "target uncertainty = 1 - clip(top1_score - top2_score, 0, 1)",
            "condition uncertainty = mean(solvent uncertainty, catalyst uncertainty)",
            "```",
            "",
            (
                "This is an interpretable ranking-separation heuristic. "
                "It is not a calibrated probability."
            ),
            "",
            (
                "Qualitative confidence combines condition-model "
                "applicability, nearest historical RXNFP similarity, "
                "and the condition-ranking uncertainty."
            ),
            "",
            "## Lightweight synthesizability risk",
            "",
            "The risk assessment combines:",
            "",
            "- RDKit SA score.",
            "- Rare Morgan fragments according to the SA fragment corpus.",
            "- Bertz molecular complexity.",
            "- Nearest historical reaction similarity, when a reaction is supplied.",
            "- Condition-ranking uncertainty, when a reaction is supplied.",
            "",
            (
                "This risk assessment provides warnings and prioritization "
                "signals only. It does not perform reaction-template "
                "enumeration, route search, reagent planning, or multi-step "
                "retrosynthesis."
            ),
            "",
            "## Applicability and scientific interpretation",
            "",
            (
                "The molecular applicability scope reproduces the Day 1 "
                "rules: valid SMILES, one fragment, at least one carbon "
                "atom, only configured common elements, and molecular "
                "weight between 50 and 1,000 inclusive."
            ),
            "",
            (
                "Out-of-scope molecules are not silently rejected. The "
                "numerical prediction is returned with an explicit "
                "reliability warning."
            ),
            "",
            (
                "Condition ranking scores are not calibrated reaction-success "
                "probabilities. RXNFP cosine similarity is not a probability "
                "of reaction success. ORD LC area percent at 280 nm is not "
                "isolated reaction yield."
            ),
            "",
            (
                "Solvent and catalyst are predicted independently and do "
                "not constitute a jointly optimized reaction condition."
            ),
            "",
            "## API",
            "",
            "The API exposes:",
            "",
            "- `GET /health`: artifact availability without loading models.",
            "- `POST /v1/predict`: molecule-only, reaction-only, or combined inference.",
            "- `GET /docs`: interactive OpenAPI documentation.",
            "- `GET /openapi.json`: machine-readable API schema.",
            "",
            "Start the local service with:",
            "",
            "```bash",
            "python scripts/serve_chempilot.py \\",
            "  --host 127.0.0.1 \\",
            "  --port 8000",
            "```",
            "",
            "## Error handling",
            "",
            "- Invalid SMILES or reaction inputs return HTTP 422.",
            "- Schema violations and unknown JSON fields return HTTP 422.",
            "- Missing, corrupt, or incompatible model artifacts return HTTP 503.",
            "- Invalid service configuration returns HTTP 500.",
            "- Unexpected exceptions return a sanitized HTTP 500 response.",
            "- Every HTTP response carries an `X-Request-ID` header.",
            "",
            "## Validation",
            "",
            "Day 6 tests cover:",
            "",
            "- SMILES standardization and invalid structures.",
            "- Descriptor and fingerprint dimensions and ordering.",
            "- Historical model-serialization alignment.",
            "- Drug-like scope and synthesis-risk thresholds.",
            "- Lazy and thread-safe model loading.",
            "- Molecule-only, reaction-only, and combined requests.",
            "- API response schemas and OpenAPI generation.",
            "- HTTP 422, 500, and 503 failure cases.",
            "- Real Uvicorn and curl boundary requests.",
            "",
            "## Reproducibility",
            "",
            "Generate the example artifacts with:",
            "",
            "```bash",
            "python scripts/generate_day6_examples.py",
            "```",
            "",
            "Run the focused Day 6 tests with:",
            "",
            "```bash",
            "python -m pytest \\",
            "  tests/test_service_base.py \\",
            "  tests/test_service_schemas.py \\",
            "  tests/test_molecule_service.py \\",
            "  tests/test_solubility_service.py \\",
            "  tests/test_model_registry.py \\",
            "  tests/test_prediction_service.py \\",
            "  tests/test_api.py \\",
            "  tests/test_openapi.py \\",
            "  -q",
            "```",
            "",
            "## Generated artifacts",
            "",
            "- `reports/day6/examples/molecule_prediction.json`",
            "- `reports/day6/examples/reaction_prediction.json`",
            "- `reports/day6/examples/example_manifest.json`",
            "- `reports/day6/day6_unified_inference_report.md`",
            "",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print(
        f"Lines: {len(lines)}"
    )


if __name__ == "__main__":
    main()