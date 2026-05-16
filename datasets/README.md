# Dataset layout

This folder contains two parallel organizations of the data used by the public code package.

## Script-compatible folders

- `../Data/`: compatibility copy for scripts that expect dataset1-style paths.
- `../Data2/`: compatibility copy for scripts that expect dataset2-style paths.

## Cleaned public folders

- `dataset1/`
  - `association3.txt`
  - `disease.txt`
  - `drug.txt`
  - `ncRNA.txt`
  - `LLM_disease_sim.txt`
  - `LLM_drug_sim.txt`
  - `LLM_rna_sim.txt`

- `dataset2/`
  - `association.txt`
  - `disease.txt`
  - `drug.txt`
  - `RNA.txt`
  - `LLM_disease_sim.txt`
  - `LLM_drug_sim.txt`
  - `LLM_rna_sim.txt`

- `metadata/`
  - entity mapping tables used in the manuscript analyses.

The same dataset1 LLM similarity matrices are also mirrored into `../Data/` so that the legacy training scripts can run without path edits.
