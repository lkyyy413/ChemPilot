# ChemPilot

ChemPilot is a reproducible molecular machine-learning project for
drug-property prediction, synthesis-feasibility assessment, and reaction
condition recommendation.

The current milestone focuses on aqueous solubility prediction using
TDC Solubility_AqSolDB.

## Current task

- Input: molecular SMILES
- Output: aqueous solubility LogS
- Unit: log10(mol/L)
- Task type: regression
- Dataset size: 9,982 compounds
- Primary benchmark metric: MAE
- Primary evaluation: TDC official scaffold test split

## Day 1 data pipeline

The implemented pipeline performs:

1. Dataset download through PyTDC
2. Raw-data snapshot and SHA-256 provenance
3. SMILES validity and duplicate auditing
4. Conservative canonical SMILES standardization
5. Salt, mixture, charge, and element flags
6. RDKit descriptor calculation
7. Exploratory data analysis
8. Random and official scaffold split generation
9. Sample-overlap and scaffold-overlap auditing

No fragments are removed and no charges are neutralized.

## Data scopes

Two data scopes are retained:

- Official benchmark scope: all 9,982 compounds
- Drug-like analysis scope: 8,721 compounds

The drug-like scope requires a valid single-fragment carbon-containing
structure, common drug-like elements, and molecular weight between
50 and 1,000 g/mol.

The drug-like flag is used for subgroup analysis and does not replace
the official benchmark.

## EDA findings

- MolLogP has the strongest monotonic relationship with LogS:
  Spearman rho = -0.7399.
- Molecular weight has Spearman rho = -0.5386 with LogS.
- The dataset contains salts, mixtures, inorganic compounds, and
  molecular-weight extremes.
- Applicability-domain evaluation is therefore necessary.

See [the EDA report](reports/eda_solubility.md).

## Split protocols

### Diagnostic random split

A label-stratified 70/10/20 random split with seed 42 is provided for
diagnostic comparison only. It is not comparable with the TDC
leaderboard because molecular scaffolds overlap across the splits.

### Official scaffold benchmark

The fixed official TDC test set contains 1,997 compounds.

Official train-validation splits are stored for seeds 1, 2, 3, 4, and
5, with seed 42 retained for development diagnostics.

A split audit identified an important characteristic of the default
seed-42 train-validation split: all 2,940 validation molecules are
acyclic and have an empty Bemis-Murcko scaffold. This creates a strong
validation-distribution shift and will be considered when selecting
models.

## Reproduce Day 1

```bash
conda env create -f environment.yml
conda activate chempilot

python scripts/download_tdc.py
python scripts/inspect_raw_data.py
python scripts/standardize_smiles.py
python scripts/generate_eda.py
python scripts/create_splits.py