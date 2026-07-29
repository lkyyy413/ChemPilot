# Day 5: Reaction Transformer and Similar-Reaction Retrieval

Generated at 2026-07-29T11:32:41.130640+00:00.

## Objective

Day 5 evaluates whether a pretrained reaction-SMILES Transformer improves solvent and catalyst recommendation over the Day 4 Morgan-fingerprint baseline. It also builds a similar-reaction retrieval system that returns structurally relevant historical precedents and their observed conditions.

The condition labels remain multi-label targets. Reported scores are ranking scores. The ORD response is LC area percent at 280 nm and is not treated or reported as isolated reaction yield.

## Reaction-sequence representation

Each input sequence is constructed as `sorted canonical reactants>>sorted canonical products`. Reagents, solvents, catalysts, atom mappings, and target labels are excluded from the Transformer input.

| Statistic | Value |
| --- | --- |
| Modeling transformations | 381 |
| Unique sequences | 381 |
| RDKit-invalid sequences | 0 |
| Lexically unmatched sequences | 0 |
| Maximum lexical tokens | 192 |
| Sequences exceeding 128 tokens | 52 (13.65%) |
| Sequences exceeding 256 tokens | 0 |

A maximum sequence length of 256 was therefore selected: it avoids truncation for all 381 modeling transformations while remaining well below the model's 512-position limit.

## RXNFP checkpoint and tokenization

The `bert_pretrained` checkpoint was extracted from the RXNFP 0.1.0 wheel. The masked-language-model checkpoint was chosen instead of the reaction-classification fine-tuned checkpoint to avoid importing source classification-task supervision.

| Checkpoint property | Value |
| --- | --- |
| Vocabulary size | 591 |
| Hidden size | 256 |
| Transformer layers | 12 |
| Attention heads | 4 |
| Maximum positions | 512 |
| Unknown tokens | 0 |
| Sequences with unknown tokens | 0 |

## Frozen Transformer features

Two 256-dimensional reaction embeddings were cached for every modeling reaction:

- `CLS`: the final hidden state of the special classification token.
- `masked_mean`: the attention-mask-aware mean of non-padding token hidden states.

Validation search compared both pooling methods and logistic-regression regularization values. Masked-mean pooling was selected for all four condition classification tasks.

## Partial fine-tuning

The embedding layer and Transformer layers 0–9 were frozen. Layers 10–11 and a new multi-label linear classifier were trained with BCE-with-logits loss. Class imbalance was handled using positive-class weights clipped at 20.

| Task | Selected epoch | Epochs completed | Valid micro AP | Valid MRR | Valid HitRate@5 |
| --- | --- | --- | --- | --- | --- |
| Transformation / Solvent | 43 | 51 | 0.4919 | 0.6435 | 0.8372 |
| Transformation / Catalyst | 21 | 29 | 0.2835 | 0.6294 | 0.8286 |
| Reaction center / Solvent | 80 | 80 | 0.4901 | 0.7490 | 0.9375 |
| Reaction center / Catalyst | 80 | 80 | 0.2391 | 0.6122 | 0.8333 |

The reaction-center solvent and catalyst runs selected epoch 80, the search boundary. Their training loss continued to decrease while validation loss increased, indicating ranking improvement accompanied by worsening probability calibration and overfitting risk.

## Final condition-classification results

All hyperparameters and epoch counts were selected using validation data. Final models were refitted on train plus validation and evaluated once on the untouched test split.

| Task | Model | Test micro AP | Test MRR | Test HitRate@5 | Test recall@5 |
| --- | --- | --- | --- | --- | --- |
| Reaction center / Catalyst | Fine-tuned RXNFP | 0.2222 | 0.5256 | 0.6939 | 0.3766 |
| Reaction center / Catalyst | Frozen RXNFP | 0.1836 | 0.5068 | 0.6327 | 0.3311 |
| Reaction center / Catalyst | Morgan + logistic | 0.2655 | 0.6727 | 0.7143 | 0.4255 |
| Reaction center / Solvent | Fine-tuned RXNFP | 0.4107 | 0.6374 | 0.9286 | 0.7062 |
| Reaction center / Solvent | Frozen RXNFP | 0.3621 | 0.6459 | 0.8929 | 0.6943 |
| Reaction center / Solvent | Morgan + logistic | 0.5276 | 0.6848 | 0.9464 | 0.7635 |
| Transformation / Catalyst | Fine-tuned RXNFP | 0.2413 | 0.5504 | 0.6944 | 0.3036 |
| Transformation / Catalyst | Frozen RXNFP | 0.2745 | 0.5307 | 0.6944 | 0.3278 |
| Transformation / Catalyst | Morgan + logistic | 0.3268 | 0.6406 | 0.8611 | 0.4584 |
| Transformation / Solvent | Fine-tuned RXNFP | 0.3954 | 0.5936 | 0.7647 | 0.6109 |
| Transformation / Solvent | Frozen RXNFP | 0.4200 | 0.6490 | 0.8627 | 0.7619 |
| Transformation / Solvent | Morgan + logistic | 0.6229 | 0.6656 | 0.8824 | 0.7668 |

Morgan + logistic achieved the highest test micro AP on 4/4 tasks. Neither frozen embeddings nor partial fine-tuning consistently improved condition classification.

This negative result is informative: with roughly 170–270 training reactions per task and highly sparse catalyst labels, the lower-capacity Morgan baseline generalizes more reliably than the pretrained Transformer.

## Similar-reaction retrieval

Exact cosine similarity search was evaluated using frozen RXNFP embeddings. Pooling was selected on train-to-validation retrieval only; test labels were not used. CLS pooling was selected with validation score 0.6605 versus 0.6559 for masked-mean pooling.

| Protocol | Index reactions | Test queries | Type Hit@5 | Type MRR@10 | Solvent recall@5 | Catalyst recall@5 | Mean nearest similarity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformation | 323 | 58 | 0.8621 | 0.7714 | 0.8601 | 0.7695 | 0.9490 |
| reaction_center | 316 | 65 | 0.8154 | 0.7344 | 0.7336 | 0.6537 | 0.9455 |

Retrieval is the strongest Day 5 Transformer result. Under the transformation split, the five nearest precedents recover solvent labels with 0.8601 recall and catalyst labels with 0.7695 recall.

The end-to-end retrieval interface accepts reactant and product SMILES, canonicalizes the reaction, generates an RXNFP CLS embedding, retrieves train-plus-validation precedents, and attaches the best historically observed condition record for each neighbor.

## Conclusions

1. The Day 4 Morgan classifier remains the preferred direct condition-ranking model.
2. Partial RXNFP fine-tuning overfits the small multi-label training set and does not improve untouched-test performance.
3. Frozen RXNFP embeddings provide strong similar-reaction retrieval and useful evidence for downstream recommendations.
4. Retrieval similarity is not a probability of reaction success, and retrieved conditions are precedents rather than guaranteed optimal conditions.
5. Solvent and catalyst predictions remain independent; joint condition ranking is not implemented.

## Generated artifacts

- `reports/day5/figures/day5_model_comparison.png`
- `reports/day5/figures/day5_fine_tuning_curves.png`
- `reports/day5/figures/day5_retrieval_metrics.png`
- `reports/day5/retrieval/retrieval_example.json`
- `reports/day5/classification/three_model_final_test_comparison.csv`
