# Known Limitations and Failure Cases

## Property prediction

- Accuracy degrades for chemistry outside the AqSolDB applicability scope, including unusual elements, salts, mixtures, extreme molecular weights, and uncommon scaffolds.
- The scaffold split has an empty-Murcko-scaffold grouping defect for many acyclic compounds.
- A numeric prediction can still be returned out of domain, but it must be accompanied by a warning.
- Exact serialized predictions can drift after retraining even with pinned top-level versions; tests should emphasize model contracts and scientifically meaningful tolerances.

## Reaction-condition recommendation

- Solvent and catalyst recommendations are independent rankings, not complete jointly optimized recipes.
- Temperature, time, base, concentration, addition order, and workup are not comprehensively optimized.
- Ranking scores are uncalibrated and must not be interpreted as reaction-success probability.
- Long-tail catalyst classes have few positives, making class-level estimates noisy.
- Unknown labels and novel chemistry can fall outside the trained vocabulary and applicability domain.

## Retrieval and historical evidence

- High embedding similarity does not establish mechanistic equivalence or successful transfer of conditions.
- The nearest reaction can share superficial structure while differing at the reactive center.
- ORD LC area percentage at 280 nm is not isolated reaction yield.

## Synthesizability assessment

- SA score and fragment rarity are heuristics.
- Complexity thresholds are configuration-dependent.
- No multistep route search, building-block availability check, protecting-group plan, cost model, or reaction-network optimization is implemented.
- The risk level is guidance for review, not a feasibility guarantee.

## Operational failures handled explicitly

- Empty or invalid SMILES.
- Missing reactants or products and malformed reaction formats.
- Missing model, feature, checkpoint, or retrieval-index artifacts.
- Invalid YAML configuration and incompatible feature dimensions.
- Out-of-domain requests and weak ranking separation.
