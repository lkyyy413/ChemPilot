# ChemPilot Three-Minute Project Introduction

## Problem

Chemistry teams often evaluate molecular properties and reaction conditions in separate tools, with limited visibility into applicability and uncertainty. ChemPilot provides a single, reproducible interface for both tasks.

## Solution

For a candidate SMILES, ChemPilot returns aqueous-solubility prediction, molecular descriptors, a Lipinski assessment, SA score, fragment rarity, complexity, and a lightweight synthesizability risk. When reactants and a target product are supplied, it also returns Top-K solvent and catalyst rankings, applicability information, qualitative confidence, and similar historical reactions.

The system is built around composable featurizer, predictor, registry, and service interfaces. YAML controls artifact paths and thresholds, while FastAPI provides structured JSON and explicit errors.

## Results

The property benchmark showed that scaffold evaluation is harder than random evaluation. The deployed scaffold XGBoost model reached about `0.8059` test MAE in LogS and outperformed the GINE ensemble in this dataset regime.

For reaction conditions, Morgan logistic regression beat frozen and partially fine-tuned RXNFP classifiers on final test micro-AP across all four tasks. The strongest result was transformation-protocol solvent prediction with micro-AP `0.6229`. RXNFP was more useful for retrieval: transformation retrieval reached reaction-type Hit@5 `0.8621`.

## Responsible scope

ChemPilot does not claim calibrated reaction-success probability or multistep retrosynthesis. It reports applicability, ranking uncertainty, similar evidence, and clear limitations. Historical ORD analytical response is not isolated yield.

## Role relevance

The project demonstrates chemical data auditing, leakage-aware splitting, descriptor and graph modeling, transformer transfer learning, retrieval, uncertainty communication, API engineering, testing, and clean-environment reproducibility—connecting modeling work to a usable scientific product.
