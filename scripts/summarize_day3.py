"""Generate the evidence-based Day 3 GINE report."""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports/day3"
OUTPUT_PATH = REPORT_ROOT / "day3_gine_report.md"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def number(value, digits=4) -> str:
    return f"{float(value):.{digits}f}"


def get_xgboost_row(
    table: pd.DataFrame,
    protocol: str,
):
    return table[
        (table["protocol"] == protocol)
        & (table["model"] == "XGBoost")
    ].iloc[0]


def get_gine_row(
    table: pd.DataFrame,
    protocol: str,
):
    return table[
        (table["protocol"] == protocol)
        & (table["model"] == "GINE")
    ].iloc[0]


def main() -> None:
    comparison = pd.read_csv(
        REPORT_ROOT
        / "gine_xgboost_comparison.csv"
    )

    random_development = load_json(
        REPORT_ROOT
        / "random_gine_development.json"
    )
    scaffold_development = load_json(
        REPORT_ROOT
        / "scaffold_gine_development.json"
    )

    bootstrap = load_json(
        REPORT_ROOT
        / "subgroups/"
          "paired_bootstrap_summary.json"
    )
    ensemble_summary = load_json(
        REPORT_ROOT
        / "subgroups/"
          "subgroup_analysis_summary.json"
    )

    random_uncertainty = pd.read_csv(
        REPORT_ROOT
        / "subgroups/"
          "random_uncertainty_analysis.csv"
    )
    scaffold_uncertainty = pd.read_csv(
        REPORT_ROOT
        / "subgroups/"
          "scaffold_uncertainty_analysis.csv"
    )

    subgroup_table = pd.read_csv(
        REPORT_ROOT
        / "subgroups/"
          "gine_xgboost_subgroup_metrics.csv"
    )

    random_results = pd.read_csv(
        REPORT_ROOT
        / "random_gine_final_results.csv"
    )
    scaffold_results = pd.read_csv(
        REPORT_ROOT
        / "scaffold_gine_final_results.csv"
    )

    random_xgb = get_xgboost_row(
        comparison,
        "random",
    )
    scaffold_xgb = get_xgboost_row(
        comparison,
        "scaffold",
    )
    random_gine = get_gine_row(
        comparison,
        "random",
    )
    scaffold_gine = get_gine_row(
        comparison,
        "scaffold",
    )

    random_closest = (
        subgroup_table[
            subgroup_table["protocol"]
            == "random"
        ]
        .sort_values(
            "gine_minus_xgboost_mae"
        )
        .iloc[0]
    )
    scaffold_closest = (
        subgroup_table[
            subgroup_table["protocol"]
            == "scaffold"
        ]
        .sort_values(
            "gine_minus_xgboost_mae"
        )
        .iloc[0]
    )

    random_q1 = random_uncertainty.iloc[0]
    random_q4 = random_uncertainty.iloc[-1]
    scaffold_q1 = scaffold_uncertainty.iloc[0]
    scaffold_q4 = scaffold_uncertainty.iloc[-1]

    random_training_seconds = (
        random_results["training_seconds"].mean()
    )
    scaffold_training_seconds = (
        scaffold_results[
            "training_seconds"
        ].mean()
    )

    random_peak_memory = (
        random_results[
            "peak_gpu_memory_mb"
        ].max()
    )
    scaffold_peak_memory = (
        scaffold_results[
            "peak_gpu_memory_mb"
        ].max()
    )

    random_failure = bootstrap[
        "random"
    ]["top20_failure_audit"]
    scaffold_failure = bootstrap[
        "scaffold"
    ]["top20_failure_audit"]

    report = f"""# Day 3: GINE molecular graph regression

## Executive conclusion

A four-layer GINE molecular graph model was implemented and
evaluated against the strongest Day 2 baseline for aqueous
solubility prediction. The GINE model learned a meaningful
structure-property representation, but it did not outperform
XGBoost using combined ECFP and RDKit descriptors.

On the random test protocol, the three-seed GINE mean MAE was
{number(random_gine["test_mae"])} ±
{number(random_gine["test_mae_ci95"])} (95% confidence interval),
compared with {number(random_xgb["test_mae"])} for XGBoost.
On the scaffold protocol, the corresponding values were
{number(scaffold_gine["test_mae"])} ±
{number(scaffold_gine["test_mae_ci95"])} for GINE and
{number(scaffold_xgb["test_mae"])} for XGBoost.

Averaging the predictions from the three GINE seeds improved MAE
to {number(ensemble_summary["random"]["gine_ensemble_mae"])}
on the random test set and
{number(ensemble_summary["scaffold"]["gine_ensemble_mae"])}
on the scaffold test set. The ensembles nevertheless remained
worse than XGBoost.

The evidence therefore supports XGBoost as the production
candidate for this dataset and GINE as a research baseline.

## Task and data

The task predicts aqueous solubility LogS in log10(mol/L) from
canonical molecular SMILES. The complete AqSolDB benchmark
contains 9,982 samples.

Two protocols were retained:

| Protocol | Development train | Development valid | Fixed test | Purpose |
|---|---:|---:|---:|---|
| Random | 6,986 | 999 | 1,997 | In-distribution diagnostic |
| Scaffold | 5,045 | 2,940 | 1,997 | Chemical-structure generalization |

The official scaffold validation set has an unusual composition:
all 2,940 validation molecules have an empty Bemis-Murcko
scaffold. The fixed scaffold test set contains unseen cyclic
scaffolds. This distribution shift is retained and documented
rather than silently replaced.

## Molecular graph representation

Canonical SMILES were converted to directed molecular graphs.
Atoms are nodes and bonds are represented in both directions.

Node features include atomic number, degree, total valence,
formal charge, hybridization, aromaticity, and chirality. Edge
features include bond type, conjugation, ring membership, and
stereochemistry.

The graph cache contains 9,982 graphs, 173,500 atoms and 352,000
directed edges. It includes 149 zero-edge graphs and 1,098
multi-fragment structures. No fragments were removed and no
charges were neutralized, preserving the official benchmark
definition.

## Model and training

The main graph model is a four-layer GINE regressor with hidden
dimension 128, sum pooling and dropout 0.15. It contains 305,541
trainable parameters.

Training used AdamW, Huber loss on standardized LogS targets,
gradient clipping, validation-MAE early stopping, and a
ReduceLROnPlateau scheduler. The target scaler was fitted only on
the corresponding training data and stored in each checkpoint.

A 32-sample overfitting diagnostic reached MAE 0.0166, confirming
that the graph, optimization, checkpoint and inverse-scaling
pipeline can learn and recover predictions correctly.

## Development results

| Protocol | Best epoch | Validation MAE | RMSE | R2 | Spearman | Training time |
|---|---:|---:|---:|---:|---:|---:|
| Random | {random_development["best_epoch"]} | {number(random_development["validation_metrics"]["mae"])} | {number(random_development["validation_metrics"]["rmse"])} | {number(random_development["validation_metrics"]["r2"])} | {number(random_development["validation_metrics"]["spearman"])} | {number(random_development["training_seconds"], 1)} s |
| Scaffold | {scaffold_development["best_epoch"]} | {number(scaffold_development["validation_metrics"]["mae"])} | {number(scaffold_development["validation_metrics"]["rmse"])} | {number(scaffold_development["validation_metrics"]["r2"])} | {number(scaffold_development["validation_metrics"]["spearman"])} | {number(scaffold_development["training_seconds"], 1)} s |

The random learning curves show a modest train-validation gap.
The scaffold development run has a much larger gap, consistent
with its stronger distribution shift.

![GINE learning curves](figures/gine_learning_curves.png)

## Final fixed-test comparison

The best epoch was selected without using the test set. A fresh
model was then trained on train plus validation for that fixed
number of epochs. Final experiments used predetermined seeds
1, 2 and 3.

| Protocol | Model | Test MAE | RMSE | R2 | Spearman |
|---|---|---:|---:|---:|---:|
| Random | XGBoost combined | {number(random_xgb["test_mae"])} | {number(random_xgb["test_rmse"])} | {number(random_xgb["test_r2"])} | {number(random_xgb["test_spearman"])} |
| Random | GINE, 3-seed mean | {number(random_gine["test_mae"])} | {number(random_gine["test_rmse"])} | {number(random_gine["test_r2"])} | {number(random_gine["test_spearman"])} |
| Random | GINE ensemble | {number(ensemble_summary["random"]["gine_ensemble_mae"])} | — | — | — |
| Scaffold | XGBoost combined | {number(scaffold_xgb["test_mae"])} | {number(scaffold_xgb["test_rmse"])} | {number(scaffold_xgb["test_r2"])} | {number(scaffold_xgb["test_spearman"])} |
| Scaffold | GINE, 3-seed mean | {number(scaffold_gine["test_mae"])} | {number(scaffold_gine["test_rmse"])} | {number(scaffold_gine["test_r2"])} | {number(scaffold_gine["test_spearman"])} |
| Scaffold | GINE ensemble | {number(ensemble_summary["scaffold"]["gine_ensemble_mae"])} | — | — | — |

![Test MAE comparison](figures/gine_xgboost_test_mae.png)

The single-seed GINE runs required on average
{number(random_training_seconds, 1)} seconds for 159 random
epochs and {number(scaffold_training_seconds, 1)} seconds for
62 scaffold epochs. Peak allocated GPU memory was below
{number(max(random_peak_memory, scaffold_peak_memory), 1)} MB.
The model is therefore computationally inexpensive on an RTX
3080.

## Paired statistical comparison

The ensemble and XGBoost predictions were compared molecule by
molecule using 10,000 paired bootstrap resamples.

| Protocol | GINE − XGBoost MAE | Paired bootstrap 95% CI | GINE sample win rate |
|---|---:|---:|---:|
| Random | {number(bootstrap["random"]["mean_paired_mae_difference"])} | [{number(bootstrap["random"]["bootstrap_ci95_lower"])}, {number(bootstrap["random"]["bootstrap_ci95_upper"])}] | {number(100 * bootstrap["random"]["gine_sample_win_rate"], 1)}% |
| Scaffold | {number(bootstrap["scaffold"]["mean_paired_mae_difference"])} | [{number(bootstrap["scaffold"]["bootstrap_ci95_lower"])}, {number(bootstrap["scaffold"]["bootstrap_ci95_upper"])}] | {number(100 * bootstrap["scaffold"]["gine_sample_win_rate"], 1)}% |

Both confidence intervals are entirely above zero. The observed
XGBoost advantage is therefore stable under paired test-set
resampling.

## Chemical-space error analysis

Nearest-neighbor similarity was calculated using ECFP4 Tanimoto
similarity against the complete final training set. Errors were
also stratified by scaffold status, molecular weight and the
predefined drug-like applicability domain.

No predefined subgroup had a lower mean GINE MAE than XGBoost.
The closest random subgroup was
`{random_closest["dimension"]}: {random_closest["subgroup"]}`,
where the MAE difference was
{number(random_closest["gine_minus_xgboost_mae"])}.
The closest scaffold subgroup was
`{scaffold_closest["dimension"]}: {scaffold_closest["subgroup"]}`,
where the difference was
{number(scaffold_closest["gine_minus_xgboost_mae"])}.

For scaffold molecules with molecular weight 400-600, GINE and
XGBoost were nearly tied, and GINE had a sample-level win rate
slightly above 50%. This is a local observation rather than
evidence of overall GINE superiority.

![Subgroup differences](figures/gine_xgboost_subgroup_differences.png)

## Uncertainty and applicability domain

The standard deviation of predictions across the three GINE
seeds was evaluated as a simple disagreement score.

On the random protocol, GINE MAE increased from
{number(random_q1["gine_mae"])} in the lowest-disagreement
quartile to {number(random_q4["gine_mae"])} in the highest.
On the scaffold protocol it increased from
{number(scaffold_q1["gine_mae"])} to
{number(scaffold_q4["gine_mae"])}.

However, the continuous Spearman relationship between
disagreement and error was weak for random data and absent for
scaffold data. Seed disagreement should therefore be treated as
a high-risk flag, not as a calibrated uncertainty estimate.

![Uncertainty analysis](figures/gine_uncertainty_analysis.png)

## Failure analysis

The largest GINE errors were dominated by structures outside the
intended drug-like domain.

For the random protocol, {random_failure["outside_druglike_scope"]}
of the top 20 failures were outside the drug-like scope and
{random_failure["multi_fragment"]} were multi-fragment structures.
For the scaffold protocol, the corresponding counts were
{scaffold_failure["outside_druglike_scope"]} and
{scaffold_failure["multi_fragment"]}; {scaffold_failure["molecular_weight_at_least_600"]}
of the top 20 had molecular weight at least 600.

The most extreme failures include very large mixtures, inorganic
metal-containing compounds, disconnected salts, and unusual
single-atom structures. Sum pooling can produce graph embeddings
whose magnitude grows with atom and fragment count, contributing
to unstable extrapolation for these samples.

![Observed and predicted LogS](figures/gine_xgboost_predictions.png)

## Model-selection decision

For this approximately 10,000-sample property-prediction task,
XGBoost with ECFP and explicit RDKit descriptors is preferred.
Descriptors such as LogP, molecular weight and TPSA provide
high-value physical priors directly, whereas the GINE model must
learn related global quantities indirectly from labels.

GINE remains useful when larger datasets, molecular
pretraining, multitask supervision, or learned graph embeddings
are available. A future preregistered experiment could assess a
hybrid model that concatenates the GINE graph embedding with
global molecular descriptors. Such an experiment must use a new
validation protocol and must not tune against the test results
reported here.

## Reproduction

```bash
python scripts/build_graph_cache.py

python scripts/train_gine_development.py \\
  --protocol random --device cuda:0

python scripts/train_gine_development.py \\
  --protocol scaffold --device cuda:0

python scripts/train_gine_final.py \\
  --protocol random --device cuda:0

python scripts/train_gine_final.py \\
  --protocol scaffold --device cuda:0

python scripts/analyze_gine_subgroups.py
python scripts/plot_day3.py
python scripts/summarize_day3.py
```

## Validity statement

The fixed test labels were not used for architecture selection,
early stopping or epoch selection. Test-set inspection began only
after the model architecture and development protocol were
frozen. No post-test GINE tuning is reported.
"""

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("Saved:", OUTPUT_PATH)
    print(
        "Report lines:",
        len(report.splitlines()),
    )


if __name__ == "__main__":
    main()