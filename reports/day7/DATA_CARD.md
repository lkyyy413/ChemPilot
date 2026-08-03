# ChemPilot Data Card

## AqSolDB property dataset

- Source: Therapeutics Data Commons, `Solubility_AqSolDB`.
- Task: aqueous solubility regression.
- Label: LogS in `log10(mol/L)`.
- Rows: 9,982.
- Inputs: molecular SMILES.

Processing preserves stereochemistry and canonicalizes SMILES without automatically removing fragments, neutralizing charges, or removing isotopes. Audit flags retain potentially problematic chemistry instead of silently deleting it.

The Day 1 applicability scope requires a valid single-fragment carbon-containing molecule, only configured common elements, and molecular weight from 50 to 1,000 Da. This scope contains 8,721 molecules and is a dataset-coverage rule, not a universal definition of drug-likeness.

## ORD reaction dataset

- Source snapshot SHA256: `78c17145099d29458960ffcb6cec7a8987efeae06b100004be2255ff28e54994`.
- Standardized experiment rows: 39,347.
- Aggregated transformation-condition rows: 34,566.
- Unique transformations: 602.
- Modeling transformations: 381.

Reaction inputs contain canonical reactants and products. Reagents, solvents, and catalysts are excluded from the reaction sequence supplied to RXNFP to reduce target leakage.

Condition labels are multilabel and strongly long-tailed, especially for catalysts. Historical analytical responses are ORD LC area percentages at 280 nm and are not treated or reported as isolated reaction yield.

## Splits and leakage controls

- Property models use a fixed official test set and separate development splits.
- Reaction models use transformation and reaction-center protocols.
- Test labels are not used for hyperparameter, pooling, epoch, or threshold selection.
- Final test evaluation is intended to occur once after development choices are locked.

## Known data limitations

- AqSolDB combines heterogeneous measurement sources and protocols.
- Duplicate chemistry and measurement noise may remain even after canonicalization.
- Empty Murcko scaffolds group many acyclic compounds and can distort scaffold splits.
- ORD labels are sparse and imbalanced; rare catalysts often have very few positives.
- Reaction records do not provide a complete negative-condition space.
- Similarity to historical data does not prove reaction success.
