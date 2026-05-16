from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


DATA_ROOT = Path(__file__).resolve().parent
OMICS_DIR = DATA_ROOT / "omics_support_outputs"
OUT_DIR = DATA_ROOT / "clinical_support_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPR_PATH = OMICS_DIR / "tcga_expression_matrix.csv"
CLINICAL_CACHE = OUT_DIR / "tcga_luad_lusc_clinical_raw.csv"
PATIENT_TABLE = OUT_DIR / "lung_tcga_patient_table.csv"
AXIS_TABLE = OUT_DIR / "lung_axis_scores.csv"
KM_RESULTS = OUT_DIR / "survival_km_results.csv"
COX_RESULTS = OUT_DIR / "cox_results.csv"
CORR_RESULTS = OUT_DIR / "patient_axis_correlations.csv"
SUMMARY_JSON = OUT_DIR / "patient_survival_summary.json"

GDC_URL = "https://api.gdc.cancer.gov/cases"
GENES = ["MALAT1", "UCA1", "STAT3", "ABCB1", "ABCC1"]


def fetch_tcga_clinical() -> pd.DataFrame:
    if CLINICAL_CACHE.exists():
        cached = pd.read_csv(CLINICAL_CACHE)
        if "cohort" in cached.columns and cached["cohort"].notna().any():
            return cached

    params = {
        "filters": json.dumps(
            {
                "op": "in",
                "content": {"field": "project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]},
            }
        ),
        "fields": ",".join(
            [
                "submitter_id",
                "project.project_id",
                "demographic.gender",
                "demographic.vital_status",
                "demographic.days_to_death",
                "demographic.age_at_index",
                "diagnoses.age_at_diagnosis",
                "diagnoses.days_to_last_follow_up",
                "diagnoses.ajcc_pathologic_stage",
                "diagnoses.primary_diagnosis",
                "diagnoses.diagnosis_is_primary_disease",
            ]
        ),
        "format": "JSON",
        "size": "2000",
    }
    response = requests.get(GDC_URL, params=params, timeout=120)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]

    rows: list[dict] = []
    for hit in hits:
        cohort = hit.get("project", {}).get("project_id", "")
        case_submitter_id = hit.get("submitter_id")
        demographic = hit.get("demographic", {}) or {}
        diagnoses = hit.get("diagnoses", []) or []
        primary = None
        for diagnosis in diagnoses:
            if diagnosis.get("diagnosis_is_primary_disease") is True:
                primary = diagnosis
                break
        if primary is None and diagnoses:
            primary = diagnoses[0]
        if primary is None:
            primary = {}

        rows.append(
            {
                "cohort": cohort.replace("TCGA-", ""),
                "case_submitter_id": case_submitter_id,
                "gender": demographic.get("gender"),
                "vital_status": demographic.get("vital_status"),
                "days_to_death": demographic.get("days_to_death"),
                "days_to_last_follow_up": primary.get("days_to_last_follow_up"),
                "age_at_index": demographic.get("age_at_index"),
                "age_at_diagnosis_days": primary.get("age_at_diagnosis"),
                "ajcc_pathologic_stage": primary.get("ajcc_pathologic_stage"),
                "primary_diagnosis": primary.get("primary_diagnosis"),
            }
        )

    clinical = pd.DataFrame(rows)
    clinical.to_csv(CLINICAL_CACHE, index=False)
    return clinical


def stage_group(value: str | float | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"not reported", "stage x", "stage 0is"}:
        return None
    match = re.search(r"Stage\s+([IVX]+)", text, re.IGNORECASE)
    if not match:
        return None
    roman = match.group(1).upper()
    if roman.startswith("I") and not roman.startswith("II") and not roman.startswith("IV"):
        return "I"
    if roman.startswith("II") and not roman.startswith("III"):
        return "II"
    if roman.startswith("III"):
        return "III"
    if roman.startswith("IV"):
        return "IV"
    return None


def build_patient_table(expr: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    expr = expr.copy()
    expr["sample_type"] = expr["sample_id"].str.split("-").str[3].str[:2]
    expr = expr[expr["sample_type"] == "01"].copy()
    expr["case_submitter_id"] = expr["sample_id"].str[:12]

    clinical = clinical.copy()
    clinical["stage_group"] = clinical["ajcc_pathologic_stage"].apply(stage_group)
    clinical["age_years"] = clinical["age_at_index"]
    age_missing = clinical["age_years"].isna()
    clinical.loc[age_missing, "age_years"] = clinical.loc[age_missing, "age_at_diagnosis_days"] / 365.25
    clinical["OS_event"] = clinical["vital_status"].astype(str).str.lower().eq("dead").astype(int)
    clinical["OS_time"] = np.where(
        clinical["OS_event"].eq(1),
        clinical["days_to_death"],
        clinical["days_to_last_follow_up"],
    )

    merged = expr.merge(clinical, on=["cohort", "case_submitter_id"], how="inner")
    merged = merged.drop_duplicates(subset=["sample_id"]).copy()
    merged = merged[merged["OS_time"].notna() & (merged["OS_time"] > 0)].copy()

    for gene in GENES:
        series = merged[gene].astype(float)
        if series.skew(skipna=True) > 2.0:
            merged[gene] = np.log1p(series)
        else:
            merged[gene] = series
        merged[f"z_{gene}"] = (merged[gene] - merged[gene].mean()) / merged[gene].std(ddof=0)

    merged["cisplatin_resistance_axis_score"] = merged[[f"z_{g}" for g in GENES]].sum(axis=1)
    merged["axis_group"] = np.where(
        merged["cisplatin_resistance_axis_score"] >= merged["cisplatin_resistance_axis_score"].median(),
        "High",
        "Low",
    )
    merged["uca1_abcc1_axis_score"] = merged[["z_UCA1", "z_ABCC1"]].sum(axis=1)
    merged["uca1_abcc1_group"] = np.where(
        merged["uca1_abcc1_axis_score"] >= merged["uca1_abcc1_axis_score"].median(),
        "High",
        "Low",
    )
    merged["uca1_abcc1_group_by_cohort"] = None
    for cohort_name in merged["cohort"].dropna().unique():
        idx = merged["cohort"] == cohort_name
        cutoff = merged.loc[idx, "uca1_abcc1_axis_score"].median()
        merged.loc[idx, "uca1_abcc1_group_by_cohort"] = np.where(
            merged.loc[idx, "uca1_abcc1_axis_score"] >= cutoff,
            "High",
            "Low",
        )
    merged["malat1_stat3_abcb1_axis_score"] = merged[["z_MALAT1", "z_STAT3", "z_ABCB1"]].sum(axis=1)
    merged["malat1_axis_group"] = np.where(
        merged["malat1_stat3_abcb1_axis_score"] >= merged["malat1_stat3_abcb1_axis_score"].median(),
        "High",
        "Low",
    )
    for gene in ["MALAT1", "UCA1"]:
        merged[f"{gene}_group"] = np.where(merged[gene] >= merged[gene].median(), "High", "Low")

    keep_cols = [
        "cohort",
        "sample_id",
        "case_submitter_id",
        "gender",
        "stage_group",
        "age_years",
        "OS_time",
        "OS_event",
        *GENES,
        "cisplatin_resistance_axis_score",
        "axis_group",
        "uca1_abcc1_axis_score",
        "uca1_abcc1_group",
        "uca1_abcc1_group_by_cohort",
        "malat1_stat3_abcb1_axis_score",
        "malat1_axis_group",
        "MALAT1_group",
        "UCA1_group",
    ]
    return merged[keep_cols].copy()


def km_one(df: pd.DataFrame, time_col: str, event_col: str, group_col: str, label: str) -> dict:
    high = df[df[group_col] == "High"].copy()
    low = df[df[group_col] == "Low"].copy()
    test = logrank_test(
        high[time_col],
        low[time_col],
        event_observed_A=high[event_col],
        event_observed_B=low[event_col],
    )
    return {
        "analysis": label,
        "n_total": int(df.shape[0]),
        "n_high": int(high.shape[0]),
        "n_low": int(low.shape[0]),
        "events_total": int(df[event_col].sum()),
        "median_high": float(high[time_col].median()),
        "median_low": float(low[time_col].median()),
        "logrank_p": float(test.p_value),
    }


def fit_univariable_cox(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    model_df = df[["OS_time", "OS_event", variable]].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="OS_time", event_col="OS_event")
    summary = cph.summary.reset_index().rename(columns={"covariate": "term"})
    summary["analysis"] = f"univariable_{variable}"
    summary["hazard_ratio"] = np.exp(summary["coef"])
    summary["ci_lower"] = np.exp(summary["coef lower 95%"])
    summary["ci_upper"] = np.exp(summary["coef upper 95%"])
    return summary


def fit_multivariable_cox(df: pd.DataFrame, score_col: str, analysis_name: str) -> pd.DataFrame:
    base = df[["OS_time", "OS_event", score_col, "age_years", "gender", "stage_group"]].copy()
    base["male"] = base["gender"].astype(str).str.lower().eq("male").astype(int)
    model_df = pd.concat(
        [
            base[["OS_time", "OS_event", score_col, "age_years", "male"]],
            pd.get_dummies(base["stage_group"], prefix="stage", drop_first=True),
        ],
        axis=1,
    ).dropna()
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="OS_time", event_col="OS_event")
    summary = cph.summary.reset_index().rename(columns={"covariate": "term"})
    summary["analysis"] = analysis_name
    summary["hazard_ratio"] = np.exp(summary["coef"])
    summary["ci_lower"] = np.exp(summary["coef lower 95%"])
    summary["ci_upper"] = np.exp(summary["coef upper 95%"])
    return summary


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    analyses = []
    pairs = [
        ("MALAT1", "STAT3"),
        ("MALAT1", "ABCB1"),
        ("UCA1", "ABCC1"),
        ("uca1_abcc1_axis_score", "UCA1"),
        ("uca1_abcc1_axis_score", "ABCC1"),
        ("cisplatin_resistance_axis_score", "STAT3"),
        ("cisplatin_resistance_axis_score", "ABCB1"),
    ]
    for cohort_name, cohort_df in [("TCGA-lung", df), ("LUAD", df[df["cohort"] == "LUAD"]), ("LUSC", df[df["cohort"] == "LUSC"])]:
        pvals = []
        rows = []
        for x, y in pairs:
            sub = cohort_df[[x, y]].dropna()
            r, p = spearmanr(sub[x], sub[y])
            rows.append(
                {
                    "cohort": cohort_name,
                    "feature_x": x,
                    "feature_y": y,
                    "n": int(sub.shape[0]),
                    "spearman_r": float(r),
                    "p_value": float(p),
                }
            )
            pvals.append(p)
        fdr = multipletests(pvals, method="fdr_bh")[1]
        for row, adj in zip(rows, fdr):
            row["fdr_p"] = float(adj)
            analyses.append(row)
    return pd.DataFrame(analyses)


def main() -> None:
    if not EXPR_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EXPR_PATH}. Run omics_support_analysis.py first to generate TCGA expression summaries."
        )

    expr = pd.read_csv(EXPR_PATH)
    clinical = fetch_tcga_clinical()
    patient_df = build_patient_table(expr, clinical)

    patient_df.to_csv(PATIENT_TABLE, index=False)
    patient_df.to_csv(AXIS_TABLE, index=False)

    km_records = [
        km_one(patient_df, "OS_time", "OS_event", "axis_group", "TCGA-lung:axis_group"),
        km_one(patient_df, "OS_time", "OS_event", "uca1_abcc1_group", "TCGA-lung:uca1_abcc1_group"),
        km_one(patient_df, "OS_time", "OS_event", "MALAT1_group", "TCGA-lung:MALAT1_group"),
        km_one(patient_df, "OS_time", "OS_event", "UCA1_group", "TCGA-lung:UCA1_group"),
        km_one(patient_df[patient_df["cohort"] == "LUAD"], "OS_time", "OS_event", "axis_group", "LUAD:axis_group"),
        km_one(patient_df[patient_df["cohort"] == "LUSC"], "OS_time", "OS_event", "axis_group", "LUSC:axis_group"),
        km_one(patient_df[patient_df["cohort"] == "LUAD"], "OS_time", "OS_event", "uca1_abcc1_group_by_cohort", "LUAD:uca1_abcc1_group_by_cohort"),
        km_one(patient_df[patient_df["cohort"] == "LUSC"], "OS_time", "OS_event", "uca1_abcc1_group_by_cohort", "LUSC:uca1_abcc1_group_by_cohort"),
    ]
    km_df = pd.DataFrame(km_records)
    km_df.to_csv(KM_RESULTS, index=False)

    cox_frames = [
        fit_univariable_cox(patient_df, "cisplatin_resistance_axis_score"),
        fit_univariable_cox(patient_df, "uca1_abcc1_axis_score"),
        fit_univariable_cox(patient_df, "MALAT1"),
        fit_univariable_cox(patient_df, "UCA1"),
        fit_multivariable_cox(patient_df, "cisplatin_resistance_axis_score", "multivariable_age_gender_stage"),
        fit_multivariable_cox(patient_df, "uca1_abcc1_axis_score", "multivariable_uca1_abcc1_axis"),
    ]
    cox_df = pd.concat(cox_frames, ignore_index=True)
    cox_df.to_csv(COX_RESULTS, index=False)

    corr_df = correlation_table(patient_df)
    corr_df.to_csv(CORR_RESULTS, index=False)

    summary = {
        "patient_count": int(len(patient_df)),
        "cohort_counts": patient_df["cohort"].value_counts().to_dict(),
        "main_km_logrank_p": float(km_df.loc[km_df["analysis"] == "TCGA-lung:uca1_abcc1_group", "logrank_p"].iloc[0]),
        "main_cox_axis_hr": float(
            cox_df.loc[
                (cox_df["analysis"] == "multivariable_uca1_abcc1_axis")
                & (cox_df["term"] == "uca1_abcc1_axis_score"),
                "hazard_ratio",
            ].iloc[0]
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
