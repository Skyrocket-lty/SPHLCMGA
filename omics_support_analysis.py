import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xenaPython as xena
from scipy.stats import mannwhitneyu, spearmanr


GENES = ["MALAT1", "UCA1", "STAT3", "ABCB1", "ABCC1"]
TCGA_GENE_PAIRS = [
    ("MALAT1", "STAT3"),
    ("MALAT1", "ABCB1"),
    ("MALAT1", "ABCC1"),
    ("UCA1", "ABCB1"),
    ("UCA1", "ABCC1"),
]

TCGA_COHORTS = {
    "LUAD": "TCGA Lung Adenocarcinoma (LUAD)",
    "LUSC": "TCGA Lung Squamous Cell Carcinoma (LUSC)",
}
TCGA_DATASETS = {
    "LUAD": "TCGA.LUAD.sampleMap/HiSeqV2",
    "LUSC": "TCGA.LUSC.sampleMap/HiSeqV2",
}

CMP_API = "https://api.cellmodelpassports.sanger.ac.uk"
DATA_ROOT = Path(__file__).resolve().parent
OUTDIR = DATA_ROOT / "omics_support_outputs"
CACHE_DIR = OUTDIR / "cmp_rnaseq_cache"


def bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = ranked[idx] * n / rank
        prev = min(prev, value)
        adjusted[idx] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def fetch_tcga_expression():
    rows = []
    for cohort_key, dataset in TCGA_DATASETS.items():
        samples = xena.dataset_samples(xena.PUBLIC_HUBS["tcgaHub"], dataset, None)
        tumor_samples = [sample for sample in samples if "-01" in sample]
        gene_payload = xena.dataset_gene_probe_avg(xena.PUBLIC_HUBS["tcgaHub"], dataset, tumor_samples, GENES)
        gene_to_scores = {
            entry["gene"]: entry["scores"][0] for entry in gene_payload
        }
        for idx, sample in enumerate(tumor_samples):
            row = {"cohort": cohort_key, "sample_id": sample}
            for gene in GENES:
                row[gene] = float(gene_to_scores[gene][idx]) if gene in gene_to_scores else np.nan
            rows.append(row)
    tcga_df = pd.DataFrame(rows)
    tcga_df.to_csv(OUTDIR / "tcga_expression_matrix.csv", index=False)
    return tcga_df


def summarize_tcga_correlations(tcga_df):
    rows = []
    for cohort_key, cohort_df in tcga_df.groupby("cohort"):
        for gene_a, gene_b in TCGA_GENE_PAIRS:
            valid = cohort_df[[gene_a, gene_b]].dropna()
            r_val, p_val = spearmanr(valid[gene_a], valid[gene_b])
            rows.append(
                {
                    "cohort": cohort_key,
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "sample_size": int(len(valid)),
                    "spearman_r": float(r_val),
                    "p_value": float(p_val),
                }
            )
    combined_df = tcga_df.copy()
    for gene_a, gene_b in TCGA_GENE_PAIRS:
        valid = combined_df[[gene_a, gene_b]].dropna()
        r_val, p_val = spearmanr(valid[gene_a], valid[gene_b])
        rows.append(
            {
                "cohort": "Combined_LUAD_LUSC",
                "gene_a": gene_a,
                "gene_b": gene_b,
                "sample_size": int(len(valid)),
                "spearman_r": float(r_val),
                "p_value": float(p_val),
            }
        )
    corr_df = pd.DataFrame(rows)
    corr_df["fdr_p"] = bh_fdr(corr_df["p_value"].tolist())
    corr_df.to_csv(OUTDIR / "tcga_correlation_results.csv", index=False)
    return corr_df


def fetch_cmp_models():
    base = f"{CMP_API}/models"
    include = "sample,sample.cancer_type,identifiers,identifiers.source"
    first = requests.get(base, params={"page[size]": 100, "page[number]": 1, "include": include}, timeout=120).json()
    total = int(first["meta"]["count"])
    pages = math.ceil(total / 100)
    all_models = []
    for page in range(1, pages + 1):
        payload = first if page == 1 else requests.get(base, params={"page[size]": 100, "page[number]": page, "include": include}, timeout=120).json()
        included = payload.get("included", [])
        by_type = {}
        for item in included:
            by_type.setdefault(item["type"], {})[str(item["id"])] = item
        source_names = {key: value["attributes"]["name"] for key, value in by_type.get("model_identifier_source", {}).items()}

        for model in payload["data"]:
            sample_rel = model["relationships"]["sample"]["data"]
            sample = by_type["sample"][sample_rel["id"]]
            cancer_rel = sample["relationships"]["cancer_type"]["data"]
            cancer_name = by_type["cancer_type"][str(cancer_rel["id"])] ["attributes"]["name"]
            identifiers = []
            for rel in model["relationships"]["identifiers"]["data"]:
                ident = by_type["model_identifier"][str(rel["id"])]
                source_id = str(ident["relationships"]["source"]["data"]["id"])
                identifiers.append({
                    "source_name": source_names.get(source_id),
                    "identifier": ident["attributes"]["identifier"],
                })
            all_models.append(
                {
                    "sidm": model["id"],
                    "model_name": model["attributes"]["names"][0] if model["attributes"]["names"] else model["id"],
                    "cancer_type": cancer_name,
                    "expression_available": bool(model["attributes"].get("expression_available")),
                    "rnaseq_available": bool(model["attributes"].get("rnaseq_available")),
                    "drugs_available": bool(model["attributes"].get("drugs_available")),
                    "identifiers": identifiers,
                }
            )
        time.sleep(0.15)
    return pd.DataFrame(all_models)


def depmap_id(identifiers):
    for item in identifiers:
        if item["source_name"] and "DepMap_ID" in item["source_name"]:
            return item["identifier"]
    return None


def fetch_model_cisplatin(sidm):
    url = f"{CMP_API}/models/{sidm}/nlmes"
    payload = requests.get(url, params={"include": "drug", "fields[drug]": "drug_name,putative_target", "page[size]": 400, "sort": "z_score"}, timeout=120).json()
    drug_names = {str(item["id"]): item["attributes"]["drug_name"] for item in payload.get("included", []) if item["type"] == "drug"}
    for row in payload.get("data", []):
        drug_id = str(row["relationships"]["drug"]["data"]["id"])
        if drug_names.get(drug_id) == "Cisplatin":
            attrs = row["attributes"]
            return {
                "ln_ic50": float(attrs["ln_ic50"]),
                "auc": float(attrs["auc"]),
                "z_score": float(attrs["z_score"]),
            }
    return None


def fetch_model_rnaseq(sidm):
    cache_path = CACHE_DIR / f"{sidm}_rnaseq.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        payload = requests.get(f"{CMP_API}/models/{sidm}/files", params={"page[size]": 50}, timeout=120).json()
        rna_files = [
            row for row in payload.get("data", [])
            if row["attributes"]["meta"].get("data_type") == "RNASeq"
            and row["attributes"]["meta"].get("level") == "processed"
        ]
        if not rna_files:
            return None
        content = requests.get(rna_files[0]["attributes"]["url"], timeout=120).content
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            with zf.open(zf.namelist()[0]) as handle:
                df = pd.read_csv(handle)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    subset = df[df["gene_symbol"].isin(GENES)].copy()
    if subset.empty:
        return None
    return {row.gene_symbol: float(row.rsem_tpm) for row in subset.itertuples(index=False)}


def build_cmp_manifest(models_df, restrict_to_lung=True):
    records = []
    for row in models_df.itertuples(index=False):
        if not row.expression_available or not row.rnaseq_available or not row.drugs_available:
            continue
        depmap = depmap_id(row.identifiers)
        if depmap is None:
            continue
        is_lung = "lung" in row.cancer_type.lower() or "bronch" in row.cancer_type.lower()
        if restrict_to_lung and not is_lung:
            continue
        drug = fetch_model_cisplatin(row.sidm)
        if drug is None:
            continue
        expr = fetch_model_rnaseq(row.sidm)
        if expr is None or any(gene not in expr for gene in ["MALAT1", "UCA1"]):
            continue
        records.append(
            {
                "sidm": row.sidm,
                "depmap_id": depmap,
                "model_name": row.model_name,
                "cancer_type": row.cancer_type,
                "analysis_scope": "lung" if is_lung else "pan-cancer",
                "ln_ic50": drug["ln_ic50"],
                "auc": drug["auc"],
                "z_score": drug["z_score"],
                **{gene: expr.get(gene, np.nan) for gene in GENES},
            }
        )
        time.sleep(0.05)
    manifest = pd.DataFrame(records)
    manifest.to_csv(OUTDIR / "cell_line_match_manifest.csv", index=False)
    return manifest


def assign_response_groups(manifest):
    lung = manifest[manifest["analysis_scope"] == "lung"].copy()
    chosen_scope = "lung"
    working = lung
    if len(lung) < 30:
        chosen_scope = "pan-cancer"
        working = manifest.copy()

    low_cut = working["ln_ic50"].quantile(1 / 3)
    high_cut = working["ln_ic50"].quantile(2 / 3)
    working = working.copy()
    working["response_group"] = "intermediate"
    working.loc[working["ln_ic50"] <= low_cut, "response_group"] = "sensitive"
    working.loc[working["ln_ic50"] >= high_cut, "response_group"] = "resistant"

    if (working["response_group"] == "sensitive").sum() < 10 or (working["response_group"] == "resistant").sum() < 10:
        if chosen_scope == "lung":
            chosen_scope = "pan-cancer"
            working = manifest.copy()
            low_cut = working["ln_ic50"].quantile(1 / 3)
            high_cut = working["ln_ic50"].quantile(2 / 3)
            working["response_group"] = "intermediate"
            working.loc[working["ln_ic50"] <= low_cut, "response_group"] = "sensitive"
            working.loc[working["ln_ic50"] >= high_cut, "response_group"] = "resistant"
    return chosen_scope, working


def summarize_ccle_gdsc(working_df):
    rows = []
    test_genes = ["MALAT1", "UCA1", "ABCB1", "ABCC1"]
    filtered = working_df[working_df["response_group"].isin(["sensitive", "resistant"])].copy()
    for gene in test_genes:
        sensitive = filtered.loc[filtered["response_group"] == "sensitive", gene].dropna().to_numpy()
        resistant = filtered.loc[filtered["response_group"] == "resistant", gene].dropna().to_numpy()
        stat, p_val = mannwhitneyu(resistant, sensitive, alternative="two-sided")
        effect = float(np.median(resistant) - np.median(sensitive))
        rows.append(
            {
                "gene": gene,
                "sensitive_n": int(len(sensitive)),
                "resistant_n": int(len(resistant)),
                "sensitive_median": float(np.median(sensitive)),
                "resistant_median": float(np.median(resistant)),
                "median_difference": effect,
                "mannwhitney_u": float(stat),
                "p_value": float(p_val),
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_p"] = bh_fdr(result["p_value"].tolist())
    result.to_csv(OUTDIR / "ccle_gdsc_cisplatin_results.csv", index=False)
    return result


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tcga_df = fetch_tcga_expression()
    tcga_corr = summarize_tcga_correlations(tcga_df)
    models_df = fetch_cmp_models()
    manifest = build_cmp_manifest(models_df, restrict_to_lung=True)
    if len(manifest) < 30:
        manifest = build_cmp_manifest(models_df, restrict_to_lung=False)
    chosen_scope, working = assign_response_groups(manifest)
    working.to_csv(OUTDIR / "cell_line_match_manifest_grouped.csv", index=False)
    ccle_results = summarize_ccle_gdsc(working)

    summary = {
        "tcga_sample_counts": tcga_df.groupby("cohort").size().to_dict(),
        "cmp_manifest_size": int(len(manifest)),
        "chosen_cmp_scope": chosen_scope,
        "cmp_group_counts": working["response_group"].value_counts().to_dict(),
        "notable_tcga_pairs": tcga_corr.sort_values("fdr_p").head(5).to_dict("records"),
        "notable_ccle_genes": ccle_results.sort_values("fdr_p").head(4).to_dict("records"),
    }
    (OUTDIR / "omics_support_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
