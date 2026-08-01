# Day 6: Unified ChemPilot Inference Service

Generated at 2026-08-01T13:33:24.299002+00:00.

## Objective

Day 6 combines the molecular-property and reaction models from Days 1–5 into one configuration-driven inference service.

A molecule SMILES request returns aqueous-solubility prediction, molecular descriptors, Lipinski drug-likeness, SA score, rare-fragment count, structural complexity, and a lightweight synthesizability-risk assessment.

When reactants and target products are supplied, the service additionally returns Top-K solvent and catalyst rankings, applicability-domain information, similar historical reactions, qualitative confidence, and a context-aware synthesis-risk assessment.

The synthesizability output is not a multi-step retrosynthesis plan and is not a guarantee that a compound or reaction is experimentally feasible.

## Architecture

```text
PredictionRequest
        |
PredictionService
        |
        +-- MoleculeStandardizer
        +-- MoleculeRiskAnalyzer
        +-- SolubilityPredictor
        +-- ReactionConditionPredictor
        +-- SimilarReactionSearch
        |
ModelRegistry -- configs/inference.yaml
        |
UnifiedPredictionResponse
```

The main composition interfaces are:

- `BaseFeaturizer`: validates feature names, dimensions, rows, and finite values.
- `BasePredictor`: defines a reusable prediction interface.
- `ModelRegistry`: resolves configuration, lazily loads models, caches instances, and reports artifact availability.
- `PredictionService`: orchestrates molecule and reaction inference and maps outputs into strict Pydantic schemas.
- `FastAPI`: exposes structured HTTP JSON endpoints and OpenAPI documentation.

## Configuration

Model paths, supported protocols, device selection, drug-like scope, and synthesis-risk thresholds are stored in `configs/inference.yaml`.

Configuration SHA256: `77d45696e6052f1b6980f466a1339c9af719ebd78afcc2d9b25a0650119cb48f`

The selected production components are:

- Solubility: Day 2 scaffold-split XGBoost with descriptor-plus-ECFP features.
- Condition recommendation: Day 4 Morgan logistic multi-label classifiers.
- Historical retrieval: Day 5 RXNFP CLS embeddings.

## Molecular inference example

Input SMILES: `CC(=O)Oc1ccccc1C(=O)O`

Canonical SMILES: `CC(=O)Oc1ccccc1C(=O)O`

| Output | Value |
|---|---:|
| Predicted LogS | -1.9036 log10(mol/L) |
| Molecular weight | 180.159 |
| MolLogP | 1.3101 |
| TPSA | 63.600 |
| Lipinski pass | True |
| SA score | 1.5800 |
| Rare fragments | 0 |
| Molecular complexity | 343.2229 |
| Synthesis-risk level | low |

The LogS model uses 2,058 features in the exact Day 2 training order: 10 global RDKit descriptors followed by 2,048 ECFP4/Morgan bits.

## Reaction inference example

Canonical reactants:

- `Brc1ccc2ncccc2c1`
- `O=S([O-])C1CC1.[Na+]`

Canonical products:

- `c1cnc2ccc(C3CC3)cc2c1`

| Output | Result |
|---|---|
| Top solvent | `SMILES:CCC(C)(C)O` (score 0.8548) |
| Top catalyst | `SMILES:C1CCC(P(C2CCCCC2)C2CCCC2)CC1.C1CCC(P(C2CCCCC2)C2CCCC2)CC1.[Fe]` (score 0.7897) |
| Nearest historical reaction type | NI COUPLING |
| Nearest RXNFP similarity | 0.9617 |
| Condition uncertainty | 0.7505 |
| Qualitative confidence | low |
| Context-aware synthesis risk | high |

## Confidence and uncertainty

For each target, the ranking uncertainty is defined from the Top-1 versus Top-2 score margin:

```text
target uncertainty = 1 - clip(top1_score - top2_score, 0, 1)
condition uncertainty = mean(solvent uncertainty, catalyst uncertainty)
```

This is an interpretable ranking-separation heuristic. It is not a calibrated probability.

Qualitative confidence combines condition-model applicability, nearest historical RXNFP similarity, and the condition-ranking uncertainty.

## Lightweight synthesizability risk

The risk assessment combines:

- RDKit SA score.
- Rare Morgan fragments according to the SA fragment corpus.
- Bertz molecular complexity.
- Nearest historical reaction similarity, when a reaction is supplied.
- Condition-ranking uncertainty, when a reaction is supplied.

This risk assessment provides warnings and prioritization signals only. It does not perform reaction-template enumeration, route search, reagent planning, or multi-step retrosynthesis.

## Applicability and scientific interpretation

The molecular applicability scope reproduces the Day 1 rules: valid SMILES, one fragment, at least one carbon atom, only configured common elements, and molecular weight between 50 and 1,000 inclusive.

Out-of-scope molecules are not silently rejected. The numerical prediction is returned with an explicit reliability warning.

Condition ranking scores are not calibrated reaction-success probabilities. RXNFP cosine similarity is not a probability of reaction success. ORD LC area percent at 280 nm is not isolated reaction yield.

Solvent and catalyst are predicted independently and do not constitute a jointly optimized reaction condition.

## API

The API exposes:

- `GET /health`: artifact availability without loading models.
- `POST /v1/predict`: molecule-only, reaction-only, or combined inference.
- `GET /docs`: interactive OpenAPI documentation.
- `GET /openapi.json`: machine-readable API schema.

Start the local service with:

```bash
python scripts/serve_chempilot.py \
  --host 127.0.0.1 \
  --port 8000
```

## Error handling

- Invalid SMILES or reaction inputs return HTTP 422.
- Schema violations and unknown JSON fields return HTTP 422.
- Missing, corrupt, or incompatible model artifacts return HTTP 503.
- Invalid service configuration returns HTTP 500.
- Unexpected exceptions return a sanitized HTTP 500 response.
- Every HTTP response carries an `X-Request-ID` header.

## Validation

Day 6 tests cover:

- SMILES standardization and invalid structures.
- Descriptor and fingerprint dimensions and ordering.
- Historical model-serialization alignment.
- Drug-like scope and synthesis-risk thresholds.
- Lazy and thread-safe model loading.
- Molecule-only, reaction-only, and combined requests.
- API response schemas and OpenAPI generation.
- HTTP 422, 500, and 503 failure cases.
- Real Uvicorn and curl boundary requests.

## Reproducibility

Generate the example artifacts with:

```bash
python scripts/generate_day6_examples.py
```

Run the focused Day 6 tests with:

```bash
python -m pytest \
  tests/test_service_base.py \
  tests/test_service_schemas.py \
  tests/test_molecule_service.py \
  tests/test_solubility_service.py \
  tests/test_model_registry.py \
  tests/test_prediction_service.py \
  tests/test_api.py \
  tests/test_openapi.py \
  -q
```

## Generated artifacts

- `reports/day6/examples/molecule_prediction.json`
- `reports/day6/examples/reaction_prediction.json`
- `reports/day6/examples/example_manifest.json`
- `reports/day6/day6_unified_inference_report.md`
