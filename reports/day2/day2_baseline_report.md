# Day 2: AqSolDB property-prediction baselines

## Objective

The objective of Day 2 was to establish reproducible baselines for aqueous solubility regression using global RDKit descriptors, ECFP4 fingerprints, and their concatenation. Mean prediction, Ridge regression, Random Forest, and XGBoost were compared under fixed random and scaffold protocols. Hyperparameters were selected using validation MAE, after which the selected configuration was refitted on train plus validation data and evaluated on the held-out test set.

## Experimental controls

All models within one split protocol used the same sample IDs, labels, canonical SMILES, and cached molecular features. Feature generation did not use labels. Scaling for Ridge descriptors was fitted inside the training pipeline. Test labels were not used to select model configurations. Random and scaffold scores are reported separately because the two protocols do not contain the same test samples and therefore should not be interpreted as a controlled one-variable comparison.

## Final results

| split_type | model | representation | test_mae | test_rmse | test_r2 | test_spearman |
| --- | --- | --- | --- | --- | --- | --- |
| random | XGBoost | combined | 0.6742 | 1.0257 | 0.8102 | 0.9032 |
| random | Random Forest | combined | 0.7120 | 1.0769 | 0.7907 | 0.8905 |
| random | XGBoost | descriptors | 0.7581 | 1.1615 | 0.7566 | 0.8756 |
| random | Random Forest | descriptors | 0.7702 | 1.1575 | 0.7582 | 0.8733 |
| random | XGBoost | ecfp | 0.9325 | 1.2957 | 0.6971 | 0.8332 |
| random | Random Forest | ecfp | 0.9873 | 1.3609 | 0.6658 | 0.8108 |
| random | Ridge | combined | 1.0201 | 1.4629 | 0.6138 | 0.8215 |
| random | Ridge | ecfp | 1.1150 | 1.4884 | 0.6002 | 0.7730 |
| random | Ridge | descriptors | 1.2438 | 1.7563 | 0.4434 | 0.7601 |
| random | Mean | none | 1.8812 | 2.3541 | -0.0000 | undefined |
| scaffold | XGBoost | combined | 0.7944 | 1.1066 | 0.7674 | 0.8559 |
| scaffold | Random Forest | combined | 0.8163 | 1.1430 | 0.7519 | 0.8460 |
| scaffold | Random Forest | descriptors | 0.8761 | 1.2262 | 0.7145 | 0.8248 |
| scaffold | XGBoost | descriptors | 0.8970 | 1.2399 | 0.7080 | 0.8170 |
| scaffold | Ridge | combined | 1.1046 | 1.6217 | 0.5005 | 0.7705 |
| scaffold | XGBoost | ecfp | 1.2508 | 1.5845 | 0.5232 | 0.6865 |
| scaffold | Ridge | descriptors | 1.2883 | 1.8393 | 0.3575 | 0.7167 |
| scaffold | Ridge | ecfp | 1.3193 | 1.6753 | 0.4670 | 0.6766 |
| scaffold | Random Forest | ecfp | 1.3999 | 1.7615 | 0.4107 | 0.6208 |
| scaffold | Mean | none | 1.8540 | 2.3825 | -0.0780 | undefined |

The strongest model under both protocols was XGBoost with combined RDKit descriptors and ECFP features. Random-split test MAE was 0.6742; scaffold-split test MAE was 0.7944. Combining global descriptors with local substructure fingerprints consistently outperformed either representation alone.

## Representation findings

RDKit descriptors alone formed a strong and computationally inexpensive baseline. ECFP alone improved on the linear descriptor model under the random protocol but transferred poorly to the scaffold protocol. The combined representation performed best because it captures both transferable physicochemical trends and local structural patterns. The poor scaffold performance of ECFP-only tree models shows that substructure memorization does not automatically provide reliable extrapolation to new chemical frameworks.

## Feature importance

LogP was the most important descriptor for both Random Forest and XGBoost, followed by molecular size, TPSA, fraction Csp3, and hydrogen-bond variables. This agrees with the exploratory negative association between lipophilicity and aqueous solubility. Gain and impurity importance describe model usage rather than causal contribution, and correlated descriptors can divide or exchange importance.

![Baseline test MAE](figures/baseline_test_mae.png)

![XGBoost descriptor importance](figures/xgboost_descriptor_importance.png)

![Prediction comparison](figures/xgboost_combined_predictions.png)

## Failure analysis

The largest errors were enriched for uncommon elements, charged dyes, salts, mixtures, highly halogenated structures, extreme LogS values, and structures outside the defined drug-like analysis scope. These samples expose the applicability boundary of a model trained on heterogeneous public solubility measurements. They should not be silently deleted from the official benchmark, but predictions for such compounds should be accompanied by an out-of-domain warning.

| split_type | sample_id | y_true | y_pred | absolute_error | in_druglike_scope |
| --- | --- | --- | --- | --- | --- |
| random | AQSOL_03643 | -8.3632 | -0.0359 | 8.3273 | False |
| random | AQSOL_00296 | -0.4755 | -7.1369 | 6.6614 | False |
| random | AQSOL_01673 | -7.7512 | -1.1695 | 6.5818 | False |
| random | AQSOL_05528 | -13.1719 | -7.4784 | 5.6935 | True |
| random | AQSOL_03168 | -1.0232 | -6.7109 | 5.6876 | False |
| scaffold | AQSOL_03564 | -9.0250 | -1.7127 | 7.3123 | False |
| scaffold | AQSOL_01266 | -6.6832 | -0.5347 | 6.1485 | False |
| scaffold | AQSOL_03168 | -1.0232 | -7.0839 | 6.0607 | False |
| scaffold | AQSOL_02273 | -0.1396 | -6.1876 | 6.0480 | False |
| scaffold | AQSOL_03488 | -8.2650 | -2.3691 | 5.8960 | True |

Detailed sample-level failures are stored in `top_failure_samples.csv`, and subgroup metrics are stored in `subgroup_error_analysis.csv`.
Negative R² values within narrow LogS bands should not be interpreted as overall model failure because restricting labels to a narrow interval substantially reduces the denominator variance used by R². MAE, RMSE, and sample count are more informative for these subgroup comparisons.

## Main conclusion

XGBoost with combined RDKit and ECFP features is the Day 2 reference baseline. It should be treated as the model that subsequent GNN and Transformer experiments must exceed under the same split and evaluation protocol. Random Forest remains a competitive, stable alternative, while descriptor-only tree models provide the best speed–interpretability trade-off.

## Known limitations

For seed 42, all 2,940 samples in the official scaffold validation split have an empty Bemis–Murcko scaffold, and the validation labels show a strong distribution shift
