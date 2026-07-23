"""Generate molecular descriptors, EDA figures, and a Markdown report."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "solubility_aqsoldb_processed.csv"
)
DESCRIPTOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "eda_descriptors.csv"
)
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_solubility.md"

DESCRIPTOR_COLUMNS = [
    "Y",
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
]


def calculate_descriptors(smiles: str) -> dict[str, Any]:
    """Calculate interpretable RDKit molecular descriptors."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return {
            "logp": None,
            "tpsa": None,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
            "ring_count": None,
            "fraction_csp3": None,
        }

    return {
        "logp": Crippen.MolLogP(molecule),
        "tpsa": rdMolDescriptors.CalcTPSA(molecule),
        "hbd": Lipinski.NumHDonors(molecule),
        "hba": Lipinski.NumHAcceptors(molecule),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "ring_count": Lipinski.RingCount(molecule),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(molecule),
    }


def save_label_distribution(dataframe: pd.DataFrame) -> None:
    """Compare LogS distributions in the complete and drug-like scopes."""
    figure, axis = plt.subplots(figsize=(8, 5))

    sns.histplot(
        dataframe["Y"],
        bins=50,
        stat="density",
        color="#4C78A8",
        alpha=0.35,
        label="Official benchmark",
        ax=axis,
    )

    sns.kdeplot(
        data=dataframe[dataframe["in_druglike_scope"]],
        x="Y",
        color="#E45756",
        linewidth=2,
        label="Drug-like scope",
        ax=axis,
    )

    axis.set_title("AqSolDB aqueous solubility distribution")
    axis.set_xlabel("LogS [log10(mol/L)]")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "label_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_molecular_weight_distribution(dataframe: pd.DataFrame) -> None:
    """Show the wide molecular-weight range using a logarithmic axis."""
    plot_data = dataframe[dataframe["molecular_weight"] > 0].copy()

    figure, axis = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=plot_data,
        x="molecular_weight",
        bins=60,
        color="#59A14F",
        alpha=0.75,
        ax=axis,
    )

    axis.set_xscale("log")
    axis.axvline(
        50,
        color="#E15759",
        linestyle="--",
        linewidth=1.5,
        label="Drug-like lower threshold",
    )
    axis.axvline(
        1000,
        color="#E15759",
        linestyle=":",
        linewidth=1.5,
        label="Drug-like upper threshold",
    )

    axis.set_title("Molecular-weight distribution")
    axis.set_xlabel("Molecular weight [g/mol, logarithmic scale]")
    axis.set_ylabel("Molecule count")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "molecular_weight_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_logp_solubility_scatter(dataframe: pd.DataFrame) -> None:
    """Visualize the relationship between lipophilicity and solubility."""
    figure, axis = plt.subplots(figsize=(8, 5.5))

    sns.scatterplot(
        data=dataframe,
        x="logp",
        y="Y",
        hue="in_druglike_scope",
        palette={
            True: "#4C78A8",
            False: "#E45756",
        },
        alpha=0.35,
        s=22,
        linewidth=0,
        ax=axis,
    )

    axis.set_title("Lipophilicity versus aqueous solubility")
    axis.set_xlabel("RDKit MolLogP")
    axis.set_ylabel("LogS [log10(mol/L)]")

    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles=handles,
        labels=[
            "Drug-like scope" if label == "True"
            else "Outside drug-like scope"
            for label in labels
        ],
        title=None,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "logp_vs_solubility.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_correlation_heatmap(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a correlation matrix for labels and descriptors."""
    correlations = dataframe[DESCRIPTOR_COLUMNS].corr(
        method="spearman"
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        correlations,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman correlation"},
        ax=axis,
    )

    axis.set_title("Descriptor and LogS correlations")
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "property_correlation.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    return correlations


def write_markdown_report(
    dataframe: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    """Write a reproducible Markdown summary of the EDA results."""
    full = dataframe
    druglike = dataframe[dataframe["in_druglike_scope"]]

    strongest_correlations = (
        correlations["Y"]
        .drop("Y")
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )

    correlation_rows = "\n".join(
        f"| {name} | {value:.4f} |"
        for name, value in strongest_correlations.items()
    )

    report = f"""# AqSolDB exploratory data analysis

## Dataset definition

- Task: aqueous solubility regression
- Input: molecular SMILES
- Label: LogS
- Unit: log10(mol/L)
- Official benchmark samples: {len(full):,}
- Drug-like analysis samples: {len(druglike):,}
- Drug-like proportion: {len(druglike) / len(full):.2%}

## Label statistics

| Scope | Mean | Median | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Official benchmark | {full["Y"].mean():.4f} | {full["Y"].median():.4f} | {full["Y"].std():.4f} | {full["Y"].min():.4f} | {full["Y"].max():.4f} |
| Drug-like scope | {druglike["Y"].mean():.4f} | {druglike["Y"].median():.4f} | {druglike["Y"].std():.4f} | {druglike["Y"].min():.4f} | {druglike["Y"].max():.4f} |

## Molecular statistics

| Property | Median | 1st percentile | 99th percentile | Maximum |
|---|---:|---:|---:|---:|
| Molecular weight | {full["molecular_weight"].median():.2f} | {full["molecular_weight"].quantile(0.01):.2f} | {full["molecular_weight"].quantile(0.99):.2f} | {full["molecular_weight"].max():.2f} |
| MolLogP | {full["logp"].median():.2f} | {full["logp"].quantile(0.01):.2f} | {full["logp"].quantile(0.99):.2f} | {full["logp"].max():.2f} |
| TPSA | {full["tpsa"].median():.2f} | {full["tpsa"].quantile(0.01):.2f} | {full["tpsa"].quantile(0.99):.2f} | {full["tpsa"].max():.2f} |

## Spearman correlation with LogS

| Descriptor | Correlation with LogS |
|---|---:|
{correlation_rows}

## Data-quality findings

- All raw SMILES were successfully parsed by RDKit.
- Canonicalization did not create duplicate structures.
- Salts, mixtures, ions, and uncommon elements remain in the official benchmark.
- No fragments were removed and no charges were neutralized.
- The drug-like scope is an analysis flag, not a replacement for the official benchmark.
- Molecular-weight and structural extremes indicate that applicability-domain checks are necessary.

## Figures

### Label distribution

![Label distribution](figures/label_distribution.png)

### Molecular-weight distribution

![Molecular-weight distribution](figures/molecular_weight_distribution.png)

### Lipophilicity versus solubility

![LogP versus LogS](figures/logp_vs_solubility.png)

### Descriptor correlations

![Property correlations](figures/property_correlation.png)
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DESCRIPTOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(INPUT_PATH)

    tqdm.pandas(desc="Calculating RDKit descriptors")
    descriptor_data = (
        dataframe["smiles_canonical"]
        .progress_apply(calculate_descriptors)
        .apply(pd.Series)
    )

    dataframe = pd.concat([dataframe, descriptor_data], axis=1)
    dataframe.to_csv(DESCRIPTOR_PATH, index=False)

    save_label_distribution(dataframe)
    save_molecular_weight_distribution(dataframe)
    save_logp_solubility_scatter(dataframe)
    correlations = save_correlation_heatmap(dataframe)
    write_markdown_report(dataframe, correlations)

    logging.info("Saved descriptor table to %s", DESCRIPTOR_PATH)
    logging.info("Saved EDA figures to %s", FIGURE_DIR)
    logging.info("Saved EDA report to %s", REPORT_PATH)


if __name__ == "__main__":
    main()