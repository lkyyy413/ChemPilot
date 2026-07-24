# ChemPilot

ChemPilot is a reproducible molecular machine-learning project for drug-property prediction, synthesis-feasibility assessment, and reaction-condition recommendation.

The current milestone establishes strong aqueous-solubility baselines on TDC Solubility_AqSolDB. Subsequent milestones will extend the same evaluation framework to graph neural networks, reaction-condition prediction, and retrieval-supported recommendations.

## Current task

- Input: molecular SMILES
- Output: aqueous solubility LogS
- Unit: log10(mol/L)
- Task type: regression
- Dataset size: 9,982 compounds
- Primary metric: mean absolute error
- Representations: RDKit descriptors, ECFP4, and their concatenation
- Models: Mean, Ridge, Random Forest, and XGBoost

## Best baseline results

| Split protocol | Model | Features | MAE | RMSE | R² | Spearman |
|---|---|---|---:|---:|---:|---:|
| Random | XGBoost | RDKit + ECFP | 0.6742 | 1.0257 | 0.8102 | 0.9032 |
| Scaffold | XGBoost | RDKit + ECFP | 0.7944 | 1.1066 | 0.7674 | 0.8559 |

Hyperparameters are selected using validation MAE. The selected configuration is then refitted on the combined training and validation data before evaluation on the held-out test set.

Random and scaffold results are reported separately because they do not use the same test samples and therefore should not be interpreted as a controlled one-variable comparison.

![Baseline comparison](reports/day2/figures/baseline_test_mae.png)

## Main findings

Combining global physicochemical descriptors with local ECFP substructure fingerprints consistently produced the strongest results. XGBoost slightly outperformed Random Forest under both protocols.

Descriptor-only tree models remained highly competitive while requiring substantially less training time. ECFP-only models transferred poorly to the scaffold protocol, suggesting that local substructure patterns alone do not provide reliable extrapolation to unfamiliar molecular frameworks.

LogP was the most important descriptor for both Random Forest and XGBoost. This agrees with the exploratory Spearman correlation of −0.7399 between MolLogP and LogS.

![Descriptor importance](reports/day2/figures/xgboost_descriptor_importance.png)

![Predicted versus observed LogS](reports/day2/figures/xgboost_combined_predictions.png)

## Failure and applicability-domain analysis

The best model performed substantially better inside the defined drug-like analysis scope:

| Split | Scope | Samples | MAE | RMSE |
|---|---|---:|---:|---:|
| Random | Drug-like | 1,738 | 0.5920 | 0.8664 |
| Random | Outside scope | 259 | 1.2256 | 1.7536 |
| Scaffold | Drug-like | 1,815 | 0.7373 | 0.9952 |
| Scaffold | Outside scope | 182 | 1.3639 | 1.8868 |

Large errors were enriched for uncommon elements, charged dyes, salts, mixtures, highly halogenated compounds, and extreme LogS values. These samples are retained in the official benchmark, but future inference should return an applicability-domain warning.

See the [complete Day 2 report](reports/day2/day2_baseline_report.md).

## Day 1: data pipeline

The data pipeline performs:

1. Dataset download through PyTDC
2. Immutable raw-data snapshot and SHA-256 provenance
3. SMILES validity and duplicate auditing
4. Conservative canonical SMILES standardization
5. Salt, mixture, charge, and uncommon-element flags
6. Exploratory analysis with RDKit descriptors
7. Random and official scaffold split generation
8. Sample-overlap and scaffold-overlap auditing

No fragments are removed and no charges are neutralized.

Two scopes are retained:

- Official benchmark scope: all 9,982 compounds
- Drug-like analysis scope: 8,721 compounds

The drug-like flag supports subgroup analysis and does not replace the official benchmark.

See the [EDA report](reports/eda_solubility.md).

## Split protocols

The diagnostic random protocol uses a label-stratified 70/10/20 split with seed 42. It is not comparable with the TDC leaderboard because molecular scaffolds overlap across its subsets.

The official TDC test set contains 1,997 compounds. A split audit identified an important characteristic of the default seed-42 development split: all 2,940 validation molecules have an empty Bemis–Murcko scaffold. The validation labels also have a strong distribution shift. This limitation is retained and documented rather than silently changing the official protocol.

## Project structure

```text
configs/                  Dataset and experiment configuration
data/                     Reproducible local data products
reports/                  Audits, result tables, figures, and analyses
scripts/                  Download, feature, training, and reporting entry points
src/chempilot/            Reusable data, feature, and evaluation modules
tests/                    Unit tests for features, alignment, and metrics
```

## Installation

```bash
conda env create -f environment.yml
conda activate chempilot
python -m pip install -e . --no-deps
```

## Reproduce Day 1

```bash
python scripts/download_tdc.py
python scripts/inspect_raw_data.py
python scripts/standardize_smiles.py
python scripts/generate_eda.py
python scripts/create_splits.py
```

## Reproduce Day 2

Build deterministic label-independent features:

```bash
python scripts/build_features.py
```

Train the linear baselines:

```bash
python scripts/train_linear_baselines.py --split all
```

Train Random Forest:

```bash
python scripts/train_random_forest.py \
  --split all \
  --representation all
```

Train XGBoost on CPU:

```bash
python scripts/train_xgboost.py \
  --split all \
  --representation all \
  --device cpu
```

A CUDA-capable environment can instead use:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_xgboost.py \
  --split all \
  --representation all \
  --device cuda:0
```

Generate consolidated tables, figures, and failure analysis:

```bash
python scripts/summarize_day2.py
```

Run tests:

```bash
python -m pytest -v
```

## Reproducibility controls

ChemPilot records dataset hashes, feature parameters, software versions, fixed sample IDs, validation searches, sample-level failure cases, and training times. Feature scaling is fitted only on training data. Test labels are not used for hyperparameter selection.

Generated datasets, feature caches, model binaries, and full sample-level prediction files are excluded from Git because they can be reconstructed using the documented commands.

## Roadmap

- Graph neural network property prediction
- Reaction-condition dataset and XGBoost baseline
- Transformer-based condition recommendation
- Similar-reaction retrieval
- Synthesis-feasibility and applicability-domain scoring
- Unified inference API and demonstration interface