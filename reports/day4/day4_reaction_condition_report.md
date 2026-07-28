# Day 4: Reaction-condition prediction

Day 4 constructs a reproducible reaction-condition benchmark from the Open Reaction Database (ORD) and trains reaction-only classifiers for solvent and catalyst recommendation.

The source data are distributed under [CC BY-SA 4.0](https://github.com/open-reaction-database/ord-data). Derived data products must preserve the applicable attribution and share-alike requirements.

## Scope and scientific interpretation

The selected d9297630 dataset contains high-throughput reaction-screening experiments. Its response variable is **LC area percent at 280 nm**, a semi-quantitative analytical response. It is not treated or reported as isolated reaction yield.

The primary Day 4 tasks are multi-label solvent and catalyst recommendation from reactant and product structures. Reagent and temperature prediction are deferred until their label and missing-value behavior are modeled explicitly.

## Dataset construction

| Quantity | Value |
| --- | --- |
| Raw standardized experiments | 39,347 |
| Unique reaction IDs | 39,347 |
| Aggregated condition pairs | 34,566 |
| Removed exact replicate rows | 4,781 |
| Unique transformations | 602 |
| Rankable transformations | 381 |
| All-zero transformations | 191 |
| Maximum replicate count | 66 |

Missing numeric conditions are retained as missing values rather than filled with global averages:

- Missing temperature: 711 (1.81%)
- Missing reaction time: 1,738 (4.42%)
- Zero-score experiments: 22,401

Exact transformation-condition replicates are aggregated, while different conditions for the same transformation are retained because they contain screening information.

## Reaction centers

Atom-mapped reaction SMILES are used to derive deterministic reaction-center signatures. Conflicting center variants within the same standardized transformation are resolved using a replicate-weighted consensus.

Reaction-center summary statistics are stored in the dataset build report and processed Parquet.

## Leakage-controlled splits

Two group-aware 70/15/15 protocols are retained. Split assignment uses group sizes and row counts only; reaction scores and condition ranks are not used.

| Protocol | Split | Condition pairs | Pair share | Transformations | Centers | Rankable transformations |
| --- | --- | --- | --- | --- | --- | --- |
| Transformation | train | 24,191 | 70.0% | 421 | 372 | 271 |
| Transformation | valid | 5,193 | 15.0% | 90 | 85 | 52 |
| Transformation | test | 5,182 | 15.0% | 91 | 87 | 58 |
| Reaction center | train | 24,174 | 69.9% | 422 | 360 | 263 |
| Reaction center | valid | 5,195 | 15.0% | 89 | 77 | 53 |
| Reaction center | test | 5,197 | 15.0% | 91 | 77 | 65 |

| Protocol | Comparison | Transformation overlap | Reaction-center overlap |
| --- | --- | --- | --- |
| Transformation | train–valid | 0 | 16 |
| Transformation | train–test | 0 | 9 |
| Transformation | valid–test | 0 | 6 |
| Reaction center | train–valid | 0 | 0 |
| Reaction center | train–test | 0 | 0 |
| Reaction center | valid–test | 0 | 0 |

Condition signatures may overlap across splits because the prediction input is the reaction structure, not a condition identifier. Exact transformations never overlap. Reaction centers additionally never overlap in the harder reaction-center protocol.

## Feature representations

| Feature matrix | Shape | Dtype | Density |
| --- | --- | --- | --- |
| Reaction combined | 34566 × 6144 | int8 | 1.905% |
| Condition combined | 34566 × 18439 | float32 | 0.201% |

The classifier input is the 6,144-dimensional reaction representation: 2,048-bit reactant Morgan fingerprint, 2,048-bit product fingerprint, and their signed difference. Condition features are built for later condition-pair scoring and are not used as inputs to the solvent/catalyst classifiers.

No label, score, condition rank, or split identity is used to construct reaction features.

## Multi-label targets

Sample unit: One sample per rankable transformation.

Positive-label policy: Union of labels appearing in top-quartile condition pairs.

Missing-condition policy: Missing solvent or catalyst is not treated as a negative or as a no-condition class.

| Protocol | Target | Classes | Minimum train frequency | Evaluable test samples | All-label-known coverage |
| --- | --- | --- | --- | --- | --- |
| Transformation | Solvent | 20 | 5 | 51 | 94.1% |
| Transformation | Catalyst | 248 | 2 | 36 | 68.4% |
| Reaction center | Solvent | 20 | 5 | 56 | 94.6% |
| Reaction center | Catalyst | 241 | 2 | 49 | 59.2% |

Vocabularies are constructed from the training split only. Unknown validation and test labels remain unknown and are not silently mapped to negatives.

## Model selection

One-vs-rest logistic regression with balanced class weights is used as the first condition classifier. The regularization parameter is selected on validation micro-average precision. Test labels are evaluated once after selection.

| Protocol | Target | Selected C | Validation micro AP | Validation HitRate@5 |
| --- | --- | --- | --- | --- |
| Transformation | Solvent | 0.10 | 0.4878 | 0.8837 |
| Transformation | Catalyst | 0.10 | 0.2118 | 0.7714 |
| Reaction center | Solvent | 0.10 | 0.6079 | 0.9167 |
| Reaction center | Catalyst | 1.00 | 0.2548 | 0.7500 |

## Final test results

| Protocol | Target | Classes | Test n | Micro AP | MRR | Hit@1 | Hit@5 | Recall@5 | Frequency Hit@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformation | Solvent | 20 | 51 | 0.6229 | 0.6656 | 0.5098 | 0.8824 | 0.7668 | 0.8627 |
| Transformation | Catalyst | 248 | 36 | 0.3268 | 0.6406 | 0.5000 | 0.8611 | 0.4584 | 0.6944 |
| Reaction center | Solvent | 20 | 56 | 0.5276 | 0.6848 | 0.5179 | 0.9464 | 0.7635 | 0.7500 |
| Reaction center | Catalyst | 241 | 49 | 0.2655 | 0.6727 | 0.6327 | 0.7143 | 0.4255 | 0.5102 |

All four models exceed their frequency baseline on HitRate@5. Solvent prediction is substantially more mature than catalyst prediction. Catalyst evaluation remains difficult because hundreds of classes are learned from only about two hundred final training transformations.

![Final Top-5 performance](figures/condition_classifier_hit_rate.png)

![Final classification metrics](figures/condition_classifier_metrics.png)

## Label and applicability coverage

| Protocol | Target | AD threshold | Complete-test in-domain rate | Evaluated in/out |
| --- | --- | --- | --- | --- |
| Transformation | Solvent | 0.2220 | 94.8% | 50 / 1 |
| Transformation | Catalyst | 0.2153 | 91.4% | 35 / 1 |
| Reaction center | Solvent | 0.2213 | 84.6% | 48 / 8 |
| Reaction center | Catalyst | 0.2160 | 81.5% | 42 / 7 |

Applicability-domain thresholds are the fifth percentile of leave-one-out nearest-neighbor Tanimoto similarity in the final train-plus-validation subset. Test labels are not used to define the thresholds.

The number of evaluated out-of-domain samples is small, especially in the transformation protocol. Domain-stratified performance is therefore descriptive and must not be interpreted as evidence that out-of-domain predictions are equally reliable.

![Coverage analysis](figures/condition_classifier_coverage.png)

## Inference interface

The `ReactionConditionPredictor` accepts reactant and product SMILES and returns:

- solvent Top-K labels;
- catalyst Top-K labels;
- uncalibrated ranking scores;
- nearest training transformation;
- Tanimoto similarity and AD threshold;
- an `in_domain` warning flag.

Example:

```bash
python scripts/predict_reaction_conditions.py \
  --reactant "Brc1ccc2ncccc2c1" \
  --reactant "O=S([O-])C1CC1.[Na+]" \
  --product "c1cnc2ccc(C3CC3)cc2c1" \
  --protocol reaction_center \
  --top-k 5
```

The solvent and catalyst models are independent. Combining their separate Top-1 outputs does not establish the best joint experimental condition.

## Reproduction

Day 4 uses the dedicated Python 3.11 `chempilot-ord` environment because the ORD schema stack differs from the PyTorch environment used for Days 1–3.

```bash
python scripts/build_ord_reaction_dataset.py
python scripts/create_ord_reaction_splits.py
python scripts/build_reaction_features.py
python scripts/audit_reaction_label_space.py
python scripts/build_reaction_label_targets.py
python scripts/search_reaction_condition_classifiers.py
# Run final test evaluation using selected C values
python scripts/analyze_reaction_applicability.py
python scripts/plot_day4.py
python scripts/summarize_day4.py
```

## Limitations and next steps

1. LC area percent is not isolated yield.
2. Solvent and catalyst recommendations are independent multi-label outputs.
3. Logistic scores are not calibrated probabilities.
4. Catalyst vocabularies are large relative to the training sample count.
5. Unknown test labels make closed-vocabulary metrics optimistic for some catalyst samples.
6. The AD threshold is a structural coverage heuristic, not a correctness guarantee.
7. The next modeling stage should rank joint condition pairs or predict the screening response using reaction and condition features.
