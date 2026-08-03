# ChemPilot Day 7 Reproducibility Report

## Scope

Day 7 rebuilt ChemPilot from clean environments and replayed the Day 1–6 workflow: data download, audit, standardization, EDA, split construction, feature generation, model training, evaluation, retrieval, and unified inference.

Two environments were required because the property workflow is pinned to Python 3.10 while `ord-schema==0.6.3` requires Python 3.11:

- `chempilot-day7-base`: Python 3.10, Days 1–3.
- `chempilot-day7-ord`: Python 3.11, Days 4–6.

## Data and preprocessing reproduction

- AqSolDB download: 9,982 rows with `Drug_ID`, `Drug`, and `Y`.
- All 9,982 SMILES parsed successfully.
- Audit flags: 1,098 multifragment molecules, 132 non-zero-charge molecules, and 873 molecules containing uncommon elements.
- Day 1 drug-like applicability scope: 8,721 of 9,982 molecules.
- Audit and standardization reports matched their historical checksums.
- EDA markdown matched exactly. PNG hashes differed across environments, while the images remained valid and the underlying tables and conclusions were unchanged; renderer and font metadata are therefore not treated as scientific outputs.
- Feature cache reproduced elementwise: 10 RDKit descriptors and 2,048 ECFP bits, for 2,058 combined features.

## Split audit

- Random split: 6,986 train, 999 validation, 1,997 test.
- The official property test set remained fixed at 1,997 molecules.
- No sample overlap was observed between train, validation, and test.
- A known scaffold-split defect remains: 2,940 acyclic molecules share an empty Murcko scaffold, and scaffold seeds 5 and 42 can therefore produce identical partitions.

## Property-model reproduction

- Ridge metrics reproduced with maximum absolute drift of approximately `1.59e-4`.
- Random Forest metrics reproduced to floating-point precision.
- XGBoost showed a maximum historical-versus-reproduced metric difference of approximately `0.0364`, despite the same declared NumPy, pandas, and XGBoost versions. This is recorded as non-bitwise reproducibility caused by training/runtime details such as parallel execution and early stopping.
- Historical scaffold-split combined-feature XGBoost test MAE was approximately `0.8059` logS.
- Day 3 GINE ensemble test MAE was `0.7926` on the random split and `0.8670` on the scaffold split; the corresponding XGBoost values were `0.6714` and `0.8059`.

## Reaction-condition and retrieval reproduction

- ORD source SHA256: `78c17145099d29458960ffcb6cec7a8987efeae06b100004be2255ff28e54994`.
- Reaction dataset: 39,347 standardized experiments, 34,566 aggregated rows, and 602 unique transformations.
- RXNFP wheel SHA256: `c5c1e818add6f34539a6b29bc680c47c9e7311e9383d1b34ce901481e34b58cf`.
- Best final condition classifier by test micro-AP was Morgan logistic regression for all four protocol/target combinations.
- Final transformation retrieval achieved reaction-type Hit@5 `0.8621`, MRR@10 `0.7714`, solvent recall@5 `0.8601`, and catalyst recall@5 `0.7695`.

Historical reaction responses are ORD LC area percentages at 280 nm. They are not treated or reported as isolated reaction yield.

## Unified inference reproduction

The Day 6 service passed 141 compatible tests after replacing a brittle exact-value assertion on a strongly out-of-domain antimony molecule with a scientific contract: the prediction must be finite and must include an applicability warning.

The service returns:

- aqueous solubility prediction;
- molecular descriptors and Lipinski assessment;
- SA score, rare-fragment and complexity signals;
- Top-K independent solvent and catalyst rankings;
- applicability-domain information;
- similar historical reactions;
- explicit uncertainty and scope warnings.

The synthesizability output is a lightweight risk assessment, not multistep retrosynthesis planning or a guarantee of synthetic feasibility.

## Reproducibility conclusion

The complete workflow is functionally reproducible. Stable scientific tables and arrays reproduce exactly or within documented numerical tolerances. Time measurements, PNG bytes, serialized model bytes, and some multithreaded training results are not expected to be bitwise identical.
