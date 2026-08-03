# Compute Resources and Reproduction Environment

## Hardware

- Host platform: Linux x86_64.
- GPUs: two NVIDIA GeForce RTX 3080 cards, 10 GB each.
- CUDA driver capability observed: CUDA 12.4.
- Storage during development: approximately 298 GB free on a 3.6 TB `/home` filesystem.

## Environments

| Environment | Python | Purpose |
| --- | ---: | --- |
| `chempilot-day7-base` | 3.10 | AqSolDB processing, classical property models, GINE |
| `chempilot-day7-ord` | 3.11 | ORD, RXNFP, reaction models, API |

Two environments are necessary because the property stack was developed on Python 3.10 while `ord-schema==0.6.3` requires Python 3.11.

## Representative software

- Base workflow: NumPy 1.26.4, pandas 2.3.3, RDKit 2023.09.6, scikit-learn 1.7.2, XGBoost 3.2.0, PyTorch 2.6.0, PyTorch Geometric 2.8.0.post1.
- ORD workflow: PyTorch 2.6.0, Transformers 4.48.3, FastAPI 0.141.1, Pydantic 2.13.4, XGBoost 3.2.0.

## Runtime observations

- Feature generation processed 9,982 molecules in seconds to minutes.
- Classical model searches ranged from seconds for descriptors to minutes for fingerprint Random Forest models.
- Day 5 partial fine-tuning tasks completed in seconds on one RTX 3080 because the reaction dataset and RXNFP encoder are small.
- Full clean reproduction is dominated by dependency installation, model searches, and graph-model training.

## Reproducibility policy

Scientific arrays, row identities, split membership, configurations, and stable metrics are the primary contracts. Wall-clock timings, PNG bytes, and serialized model bytes are environment-dependent and are not required to match bitwise.
