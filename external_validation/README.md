# External Validation Protocol

This directory contains the frozen external-validation workflow used to score post-benchmark ncRNA-drug-disease triplets without retraining.

## Scope

The manuscript reports two related external cohorts under the same frozen-scoring protocol:

- a **strict 2025-2026 primary cohort** used as the main temporal transfer analysis;
- a broader **2024-2026 pooled-support cohort** used as supplementary support.

The goal of this workflow is to test whether a benchmark-trained SPHLCMGA checkpoint can prioritize newly curated positive triplets beyond the original benchmark window while keeping the model fixed.

## Files

- `external_validation_template.csv`: input schema for newly curated external cases.
- `prepare_external_validation.py`: maps external triplets into the benchmark entity spaces, removes benchmark duplicates, annotates time-held-out subsets, and reports which rows are eligible for frozen scoring.
- `run_external_validation.py`: runs frozen-model scoring, exports ranked positive-case predictions, and summarizes confidence, top-k, and year-stratified performance.
- `external_validation_2025_2026_curated_retmax1200.csv`: strict curated cohort aligned with the manuscript's primary temporal-transfer analysis.
- `external_validation_2024_2026_curated_retmax1200.csv`: broader pooled-support curated cohort.
- `external_validation_2025_2026_pubmed_retmax1200.csv`: post-benchmark PubMed candidate pool used during curation.

## Recommended workflow

1. Curate external triplets from post-benchmark literature or independent databases.
2. Fill `external_validation_template.csv` with one row per validated triplet.
3. Run `prepare_external_validation.py` with the benchmark association files and the entity mapping spreadsheets.
4. Use only rows marked as `fully_mapped=True` and `already_in_benchmark=False` for frozen scoring.
5. Keep the SPHLCMGA checkpoint fixed. Do not retrain or fine-tune on the external cohort.
6. Report the strict 2025-2026 subset as the primary temporal-transfer cohort, and optionally retain the broader 2024-2026 cohort as pooled support.

## Expected external CSV columns

- `ncrna_name`
- `drug_name`
- `disease_name`
- `label`: use `resistance` or `sensitivity`
- `source_year`: strongly recommended for time-held-out subset definitions
- `source_id`: PMID, DOI, database accession, or other identifier
- `source_name`: source title or resource name when available
- `evidence_note`: short free-text note describing the supporting evidence

## Example preparation command

```bash
python prepare_external_validation.py   --external-csv external_validation_2025_2026_curated_retmax1200.csv   --dataset1-assoc ..\Data\association3.txt   --dataset2-assoc ..\Data2\association.txt   --dataset1-mapping ..\datasets\metadata\entity_mapping-dataset1.xlsx   --dataset2-mapping ..\datasets\metadata\entity_mapping-dataset2.xlsx   --outdir prepared_outputs_2025_2026
```

## Example frozen-scoring command

```bash
python run_external_validation.py   --prepared-csv prepared_outputs_2025_2026\dataset2_time_heldout_2025plus.csv   --data-dir ..\Data2   --assoc-path ..\Data2\association.txt   --checkpoint ..\checkpoints\best_model_fold_2.pth   --outdir results_dataset2_2025_2026   --dataset-name dataset2
```

## Key output files

Preparation outputs:

- `dataset1_prepared.csv`
- `dataset2_prepared.csv`
- `dataset2_all_external_positive.csv`
- `dataset2_time_heldout_2025plus.csv`
- `dataset2_time_heldout_2026only.csv`
- `mapping_report.json`
- `unmapped_name_suggestions.csv`

Frozen-scoring outputs:

- `external_positive_results.csv`
- `external_positive_subset_metrics.csv`
- `external_positive_ranked.csv`
- `external_positive_confidence_summary.csv`
- `external_positive_topk_hits.csv`
- `external_positive_year_summary.csv`
- `external_positive_year_label_summary.csv`
- `external_negative_sampling_results.csv`
- `external_eval_summary.json`

## Evaluation rule

Use the best internally selected checkpoint as a frozen model.

- If external positives and matched negatives are both available, report AUC and AUPR.
- If only positive literature cases are available, report case-level accuracy, ranked confidence summaries, and top-k hit rates.
- Always separate benchmark-overlap rows from truly novel rows.
- Treat this analysis as pilot evidence of temporal transfer without retraining rather than as a replacement for benchmark evaluation.
