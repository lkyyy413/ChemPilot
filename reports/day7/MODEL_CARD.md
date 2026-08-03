# ChemPilot Model Card

## Intended use

ChemPilot is a portfolio-scale decision-support system for early molecular and reaction assessment. It supports screening, model comparison, applicability checks, and retrieval of relevant historical examples. It is not a replacement for expert review or experimental validation.

## Components

### Solubility

The deployed property predictor is a scaffold-split XGBoost regressor using 10 RDKit descriptors plus a 2,048-bit ECFP4 fingerprint. The historical scaffold test MAE is approximately `0.8059` logS.

### Reaction conditions

Independent multilabel logistic classifiers rank solvent and catalyst labels from Morgan reaction features. Transformer alternatives include frozen RXNFP embeddings and partial fine-tuning of the last two encoder layers. Morgan logistic regression achieved the best final test micro-AP in all four reported protocol/target tasks.

| Protocol | Target | Test micro-AP | HitRate@5 |
| --- | --- | ---: | ---: |
| Reaction center | Catalyst | 0.2655 | 0.7143 |
| Reaction center | Solvent | 0.5276 | 0.9464 |
| Transformation | Catalyst | 0.3268 | 0.8611 |
| Transformation | Solvent | 0.6229 | 0.8824 |

Scores are ranking scores, not calibrated reaction-success probabilities. Solvent and catalyst lists are predicted independently and are not jointly optimized conditions.

### Similar-reaction retrieval

RXNFP embeddings and cosine similarity retrieve historical reactions. Final transformation-protocol reaction-type Hit@5 is `0.8621`; reaction-center Hit@5 is `0.8154`.

### Synthesizability risk

The service combines SA score, rare fragments, structural complexity, historical similarity, and condition-ranking uncertainty. This is a lightweight risk summary, not multistep retrosynthesis planning.

## Applicability and uncertainty

The system reports molecular scope warnings, reaction nearest-neighbor similarity, condition applicability, ranking separation, and qualitative confidence. These signals describe evidence coverage and uncertainty; they do not guarantee correctness.

## Evaluation cautions

- Property results depend on split design; random splits are more optimistic than scaffold splits.
- Catalyst metrics are unstable because the label space is long-tailed.
- Test metrics are aggregate summaries and should be read together with subgroup and failure analyses.
- Historical analytical response is not isolated yield.
