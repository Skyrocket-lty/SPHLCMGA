# SPHLCMGA

This repository provides the public code and dataset package for **SPHLCMGA** (Signed Perturbation Hypergraph Learning with Cross-Modal Gated Attention), a multimodal framework for decoding context-dependent **ncRNA-drug response programs across diseases**. The central motivation is that the same ncRNA-drug pair can indicate **sensitivity** in one disease context and **resistance** in another, making static binary association models insufficient for pharmacogenomic response modeling. SPHLCMGA addresses this setting through disease-conditioned ternary prediction over **sensitivity**, **resistance**, and **non-association** states, together with downstream interpretability and support analyses.

Although instantiated here in the ncRNA-drug-disease setting, the broader computational problem is **context-dependent signed ternary relation learning**, in which the same entity pair may adopt opposite functional meanings across contexts and therefore cannot be faithfully represented by a static binary association.

## Model overview

The publication workflow figure is provided in PDF format:

- `assets/figures/SPHLCMGA_framework.pdf`

## Repository layout

```text
SPHLCMGA_GitHub/
|-- main.py                         # lightweight task dispatcher
|-- dataset1_maintest.py           # dataset1 main training/evaluation script
|-- dataset2_maintest.py           # dataset2 main training/evaluation script
|-- cold_start_runner.py           # entity cold-start evaluation
|-- interpretability_runner.py     # HCMG / SP-HGNN interpretability analysis
|-- dataset2_knockout_analysis.py  # in silico knockout analysis
|-- omics_support_analysis.py      # orthogonal TCGA / cell-line molecular support
|-- patient_survival_analysis.py   # TCGA lung patient-level survival and Cox analysis
|-- polarity_switch_hcmg_analysis.py
|-- polarity_switch_scrna_analysis.py
|-- model.py / maintest.py         # compatibility shims for generic imports
|-- Data/                          # script-compatible dataset1-style folder
|-- Data2/                         # script-compatible dataset2-style folder
|-- datasets/                      # cleaned public data organization
|-- checkpoints/                   # pretrained checkpoint(s)
|-- external_validation/           # literature curation and frozen scoring scripts
|-- assets/                        # model workflow figure (PDF)
|-- requirements.txt
`-- .gitignore
```


## Environment

Recommended Python version: **3.10+**.

Install the common Python dependencies first:

```bash
pip install -r requirements.txt
```

For the PyTorch Geometric stack, install versions that match your local CUDA / PyTorch environment. If needed, install `torch`, `torch-scatter`, and `torch-geometric` from the official PyG instructions before running the model scripts.

## What this repository supports

This public snapshot is organized to support the main analyses reported in the manuscript:

- benchmark ternary prediction on dataset1 and dataset2
- entity cold-start evaluation
- HCMG / signed-hypergraph interpretability analysis
- in silico knockout analysis
- orthogonal omics support and patient-level survival support
- frozen post-benchmark literature transfer
- pathway/context large-context extension

Small example outputs are bundled where they help document the analysis structure. Newly generated runtime outputs are ignored by Git through `.gitignore`.

## Quick start

### 1. Main benchmark evaluation

```bash
python main.py --task dataset2_cv
```

This is the default entry point for reproducing the main dataset2 cross-validation benchmark.

### 2. Cold-start evaluation

```bash
python main.py --task cold_start -- --dataset dataset2 --split-axis disease
```

Change `--split-axis` to `drug` or `rna` for the other cold-start settings.

### 3. Interpretability analysis

```bash
python main.py --task interpretability -- --dataset dataset2 --project-root . --output-dir results/interpretability
```

### 4. In silico knockout analysis

```bash
python main.py --task knockout -- --data-dir Data2 --checkpoint checkpoints/best_model_fold_2.pth --output-dir results/knockout
```

### 5. Orthogonal omics support

```bash
python main.py --task omics_support
```

This generates `omics_support_outputs/tcga_expression_matrix.csv`, which is also used by the patient-level survival workflow. These output files are runtime products and are not intended to be version-controlled.

### 6. Patient-level survival support

```bash
python main.py --task patient_survival
```

Outputs are written to `clinical_support_outputs/`, including Kaplan-Meier summaries, Cox regression tables, and correlation summaries for the TCGA lung cohort.

### 7. External validation workflow

Prepare the strict 2025-2026 cohort:

```bash
python main.py --task external_validation_prepare -- --external-csv external_validation/external_validation_2025_2026_curated_retmax1200.csv --dataset1-assoc Data/association3.txt --dataset2-assoc Data2/association.txt --dataset1-mapping datasets/metadata/entity_mapping-dataset1.xlsx --dataset2-mapping datasets/metadata/entity_mapping-dataset2.xlsx --outdir external_validation/prepared_outputs_2025_2026
```

Run frozen-model scoring:

```bash
python main.py --task external_validation_run -- --prepared-csv external_validation/prepared_outputs_2025_2026/dataset2_time_heldout_2025plus.csv --data-dir Data2 --assoc-path Data2/association.txt --checkpoint checkpoints/best_model_fold_2.pth --outdir external_validation/results_dataset2_2025_2026 --dataset-name dataset2
```

The manuscript uses the strict **2025-2026** cohort as the primary temporal-transfer analysis and retains a broader **2024-2026** pooled-support cohort under the same frozen-scoring protocol.

### 8. Large-context extension

The pathway/context extension used for the manuscript's extended-dataset analyses is provided under `large_context/`. See `large_context/README.md` for matched pathway-context studies, differential correction analysis, prior-noise robustness, and evidence-tracing workflows.

## Data notes

- `Data2/` is the most complete script-ready dataset folder in this public package and includes the processed ncRNADrug matrices plus one pretrained checkpoint for demonstration.
- `Data/` contains the dataset1 low-level matrices and the dataset1 LLM similarity matrices required by the training scripts.
- The cleaned copies under `datasets/` mirror the same content in a more readable structure.
- `datasets/metadata/` contains the entity mapping tables used by the cold-start and external-validation workflows.
- `checkpoints/` contains the pretrained demonstration checkpoint used by the public external-validation and knockout examples.

See `datasets/README.md` for file-level details.

## Reproducibility notes

- The repository includes the scripts and lightweight resources needed to rerun the main benchmark, cold-start, interpretability, knockout, external-validation, and large-context workflows.
- Some auxiliary analyses depend on remote data services or precomputed intermediate files generated by earlier steps in the pipeline.
- The bundled example outputs under `large_context/analysis_examples/` and `external_validation/prepared_outputs_example/` are included to make the workflow structure transparent without requiring every full run from scratch.

## Citation

If you use this code, please cite the corresponding manuscript once the final bibliographic information is available.
