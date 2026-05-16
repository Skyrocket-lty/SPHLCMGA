import argparse
from pathlib import Path

import pandas as pd


EXCLUDE_PMIDS = {
    "41059755",  # Retraction note
    "40613620",  # Expression of concern
}

EXCLUDE_ROWS = {
    ("38928444", "LINC00662", "Fluorouracil", "Gallbladder Cancer"),  # title does not support extracted drug relation
    ("38626369", "PCAT6", "Doxorubicin", "Prostate Cancer"),  # false positive from abstract-level generic match
}

MANUAL_LABEL_OVERRIDES = {
    ("41435957", "HOTAIR", "Doxorubicin", "diffuse large B-cell lymphoma"): "resistance",
    ("41194501", "H19", "Cisplatin", "lung adenocarcinoma"): "resistance",
    ("40920672", "LINC00511", "Cisplatin", "lung adenocarcinoma"): "resistance",
    ("40309012", "GAS5", "Cisplatin", "Squamous Cell Carcinoma"): "sensitivity",
    ("40089944", "CCAT1", "Doxorubicin", "Lung Cancer"): "resistance",
    ("39733221", "PGM5-AS1", "Cisplatin", "cervical cancer"): "sensitivity",
    ("39209180", "PVT1", "Paclitaxel", "Gastric Cancer"): "resistance",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Apply manual curation rules to mined external validation candidates.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    df = df[~df["source_id"].astype(str).isin(EXCLUDE_PMIDS)].copy()
    df = df[df["source_year"].astype(str).str.extract(r"(\d{4})")[0].fillna("0").astype(int) <= 2025].copy()

    keep_mask = []
    for row in df.itertuples(index=False):
        key = (str(row.source_id), row.ncrna_name, row.drug_name, row.disease_name)
        keep_mask.append(key not in EXCLUDE_ROWS)
    df = df[pd.Series(keep_mask, index=df.index)].copy()

    for idx, row in df.iterrows():
        key = (str(row["source_id"]), row["ncrna_name"], row["drug_name"], row["disease_name"])
        if key in MANUAL_LABEL_OVERRIDES:
            df.at[idx, "label"] = MANUAL_LABEL_OVERRIDES[key]

    df = df.drop_duplicates(subset=["source_id", "ncrna_name", "drug_name", "disease_name", "label"])
    df = df.sort_values(["source_year", "source_id", "ncrna_name", "drug_name", "disease_name"]).reset_index(drop=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
