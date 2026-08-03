# ChemPilot Eight-Minute Project Introduction

## 1. Problem and product goal

Early chemistry decisions combine property questions—such as aqueous solubility—with reaction questions—such as plausible solvent and catalyst choices. ChemPilot turns these into one auditable workflow while making uncertainty and applicability visible.

## 2. Data audit

The property side uses 9,982 AqSolDB records. All SMILES parsed, but the audit identified 1,098 multifragment molecules, 132 charged molecules, and 873 molecules with uncommon elements. Rather than deleting them silently, the pipeline preserves them and records scope flags. The configured drug-like applicability scope contains 8,721 molecules.

The reaction side uses an ORD snapshot with 39,347 standardized experiments, 34,566 aggregated rows, and 602 unique transformations. Conditions are sparse multilabel targets, and catalysts are particularly long-tailed.

## 3. Features, models, and selection

Property features combine 10 interpretable RDKit descriptors with 2,048 ECFP4 bits. Ridge, Random Forest, XGBoost, and GINE models were compared. Scaffold XGBoost was selected for deployment because it gave the strongest robust result among tested models, around `0.8059` test MAE.

Reaction models compare Morgan logistic regression, frozen RXNFP embeddings, and partial RXNFP fine-tuning. Development choices use validation data only; final test evaluation occurs after pooling, regularization, and epoch choices are locked. Morgan logistic regression achieved the best final test micro-AP across solvent and catalyst tasks.

RXNFP embeddings add a complementary retrieval function. Transformation-protocol retrieval achieved Hit@5 `0.8621`, making historical examples useful evidence even when transformer classification was not the strongest predictor.

## 4. Split defects and applicability

Random property splits are optimistic because close analogs cross partitions. Scaffold splits are harder, but also contain a known defect: acyclic compounds can share an empty Murcko scaffold, placing 2,940 molecules into one group and causing some seeds to produce identical partitions.

ChemPilot therefore reports molecular scope warnings and reaction nearest-neighbor coverage rather than presenting every prediction as equally reliable. Ranking scores remain uncalibrated.

## 5. Unified inference architecture

The Day 6 service uses composable featurizer and predictor contracts, a lazy-loading model registry, and a unified prediction service. YAML controls model paths and thresholds. FastAPI exposes health and prediction endpoints with structured request and error schemas.

A molecule request returns LogS, descriptors, Lipinski rules, SA score, rare fragments, complexity, and risk. A reaction request adds independent Top-K solvent and catalyst rankings, applicability, similar reactions, confidence heuristics, and warnings.

## 6. Limitations and failure cases

The property data combines heterogeneous experimental sources. Long-tail condition labels limit class-level stability. Similarity does not establish mechanistic equivalence. Historical LC area percentage at 280 nm is not isolated yield. The synthesis assessment is not a multistep retrosynthesis planner and does not model availability, cost, protecting groups, or route networks.

Explicit errors cover invalid SMILES, malformed reactions, missing artifacts, configuration failures, and feature-dimension mismatches. Out-of-domain requests may return a number only with a warning.

## 7. Reproducibility and resources

Day 7 rebuilt the workflow in fresh Python 3.10 and 3.11 environments. Data tables, split identities, and feature arrays reproduced exactly. Random Forest reproduced to floating-point precision; Ridge showed only very small drift. XGBoost was functionally but not bitwise reproducible, with documented metric drift. PNG hashes and wall-clock times were treated as environment-dependent.

Training used two RTX 3080 GPUs, although most individual tasks require only one GPU or CPU execution.

## 8. Next experimental loop

The next iteration should repair acyclic scaffold grouping, add repeated grouped splits with confidence intervals, calibrate condition scores, improve rare-label modeling, rank joint condition sets, and connect retrieval evidence to expert feedback. A later route-planning module should be evaluated separately rather than implied by the current SA-based risk score.

## Closing

ChemPilot demonstrates an end-to-end scientific ML workflow: data audit, leakage-aware evaluation, model comparison, transfer learning, retrieval, uncertainty communication, software interfaces, testing, and reproducible delivery. The key design choice is not merely predicting—it is showing when the evidence is weak and what the output does and does not mean.
