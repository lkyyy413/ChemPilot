# ChemPilot 1–2 Minute Demo Script

## 0:00–0:15 — Problem

“ChemPilot unifies two early chemistry decisions: estimating whether a candidate molecule has usable aqueous solubility and suggesting plausible conditions for a specified reaction.”

## 0:15–0:40 — Molecule request

Open the API documentation and submit a molecule-only request with aspirin:

```json
{"molecule_smiles":"CC(=O)Oc1ccccc1C(=O)O"}
```

Point out the canonical SMILES, predicted LogS, descriptors, Lipinski result, SA score, complexity, and lightweight synthesis-risk interpretation.

## 0:40–1:15 — Reaction request

Submit a reaction containing reactants and a target product. Show Top-K solvent and catalyst rankings, applicability-domain similarity, similar historical reactions, and qualitative confidence.

Say explicitly: “The condition scores are rankings rather than calibrated success probabilities, and solvent and catalyst are predicted independently.”

## 1:15–1:35 — Responsible interpretation

Highlight the warnings: historical analytical response is not isolated yield, and synthesizability is a lightweight risk assessment rather than multistep retrosynthesis.

## 1:35–1:50 — Engineering close

“The service is driven by YAML configuration, lazy-loads model artifacts, returns structured errors, and is covered by tests for standardization, feature dimensions, serialization, API contracts, and typical failures.”
