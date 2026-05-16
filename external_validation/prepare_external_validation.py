import argparse
import json
import re
from difflib import get_close_matches
from pathlib import Path

import pandas as pd


REQUIRED_EXTERNAL_COLUMNS = [
    "ncrna_name",
    "drug_name",
    "disease_name",
    "label",
]

OPTIONAL_EXTERNAL_COLUMNS = [
    "source_year",
    "source_id",
    "source_name",
    "source_type",
    "source_date",
    "disease_context_note",
    "evidence_level",
    "evidence_note",
]


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compact_key(value):
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def normalize_label(value):
    text = normalize_text(value)
    if text in {"1", "resistance", "resistant"}:
        return 1
    if text in {"2", "sensitivity", "sensitive"}:
        return 2
    raise ValueError(f"Unsupported label value: {value!r}")


def infer_column(df, candidates):
    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    for col in df.columns:
        low = col.lower()
        if any(token in low for token in candidates):
            return col
    raise KeyError(f"Could not infer a column from {candidates}")


def normalize_entity_type(value):
    text = normalize_text(value)
    if "drug" in text or "compound" in text:
        return "drug"
    if "disease" in text or "cancer" in text or "tumor" in text or "tumour" in text:
        return "disease"
    return "ncrna"


def load_mapping(mapping_path):
    df = pd.read_excel(mapping_path)
    id_col = infer_column(df, ["id", "entity_id", "mapped_id"])
    name_col = infer_column(df, ["name", "entity_name"])
    type_col = infer_column(df, ["type", "entity_type"])

    mapping = df[[id_col, name_col, type_col]].copy()
    mapping.columns = ["entity_id", "entity_name", "entity_type"]
    mapping["entity_type"] = mapping["entity_type"].map(normalize_entity_type)
    mapping["entity_name"] = mapping["entity_name"].astype(str).str.strip()
    mapping["compact_key"] = mapping["entity_name"].map(compact_key)
    mapping = mapping[mapping["compact_key"] != ""].drop_duplicates(["entity_type", "compact_key"])
    return mapping


def build_vocab(mapping_df):
    vocab = {"ncrna": {}, "drug": {}, "disease": {}}
    names = {"ncrna": [], "drug": [], "disease": []}
    for row in mapping_df.itertuples(index=False):
        vocab[row.entity_type][row.compact_key] = {
            "entity_id": row.entity_id,
            "entity_name": row.entity_name,
        }
        names[row.entity_type].append(row.entity_name)
    return vocab, names


def load_known_triplets(assoc_path, mapping_df):
    assoc = pd.read_csv(assoc_path, sep="\t", header=None)
    if assoc.shape[1] < 4:
        raise ValueError(f"Association file has fewer than 4 columns: {assoc_path}")

    assoc = assoc.iloc[:, :4].copy()
    assoc.columns = ["ncrna_id", "drug_id", "disease_id", "label"]
    assoc["label"] = assoc["label"].map(normalize_label)

    ncrna_map = mapping_df[mapping_df["entity_type"] == "ncrna"][["entity_id", "entity_name"]].rename(
        columns={"entity_id": "ncrna_id", "entity_name": "ncrna_name"}
    )
    drug_map = mapping_df[mapping_df["entity_type"] == "drug"][["entity_id", "entity_name"]].rename(
        columns={"entity_id": "drug_id", "entity_name": "drug_name"}
    )
    disease_map = mapping_df[mapping_df["entity_type"] == "disease"][["entity_id", "entity_name"]].rename(
        columns={"entity_id": "disease_id", "entity_name": "disease_name"}
    )

    assoc = assoc.merge(ncrna_map, on="ncrna_id", how="left")
    assoc = assoc.merge(drug_map, on="drug_id", how="left")
    assoc = assoc.merge(disease_map, on="disease_id", how="left")
    assoc = assoc.dropna(subset=["ncrna_name", "drug_name", "disease_name"])

    triplets = set()
    for row in assoc.itertuples(index=False):
        triplets.add(
            (
                compact_key(row.ncrna_name),
                compact_key(row.drug_name),
                compact_key(row.disease_name),
            )
        )
    return triplets


def suggest_name(term, entity_type, name_bank, limit=3):
    if not term:
        return []
    suggestions = get_close_matches(term, name_bank[entity_type], n=limit, cutoff=0.6)
    if suggestions:
        return suggestions
    compact_to_name = {compact_key(name): name for name in name_bank[entity_type]}
    suggestions = get_close_matches(compact_key(term), list(compact_to_name.keys()), n=limit, cutoff=0.6)
    return [compact_to_name[key] for key in suggestions]


def map_external_rows(external_df, vocab, name_bank, known_triplets, dataset_name):
    rows = []
    suggestions = []
    for idx, row in external_df.iterrows():
        mapped = {
            "row_index": int(idx),
            "dataset": dataset_name,
            "ncrna_name": row["ncrna_name"],
            "drug_name": row["drug_name"],
            "disease_name": row["disease_name"],
            "label": row["label"],
            "label_id": row["label_id"],
        }
        for col in OPTIONAL_EXTERNAL_COLUMNS:
            mapped[col] = row[col] if col in row.index else None
        missing_types = []
        for entity_type, source_col in [("ncrna", "ncrna_name"), ("drug", "drug_name"), ("disease", "disease_name")]:
            key = compact_key(row[source_col])
            hit = vocab[entity_type].get(key)
            mapped[f"{entity_type}_matched"] = hit is not None
            mapped[f"{entity_type}_entity_id"] = None if hit is None else hit["entity_id"]
            mapped[f"{entity_type}_canonical_name"] = None if hit is None else hit["entity_name"]
            if hit is None:
                missing_types.append(entity_type)
                suggestions.append(
                    {
                        "row_index": int(idx),
                        "dataset": dataset_name,
                        "entity_type": entity_type,
                        "query": row[source_col],
                        "suggestions": " | ".join(suggest_name(row[source_col], entity_type, name_bank)),
                    }
                )

        mapped["fully_mapped"] = len(missing_types) == 0
        if mapped["fully_mapped"]:
            triplet_key = (
                compact_key(mapped["ncrna_canonical_name"]),
                compact_key(mapped["drug_canonical_name"]),
                compact_key(mapped["disease_canonical_name"]),
            )
            mapped["already_in_benchmark"] = triplet_key in known_triplets
        else:
            mapped["already_in_benchmark"] = False
        mapped["eligible_for_frozen_scoring"] = mapped["fully_mapped"] and not mapped["already_in_benchmark"]
        source_year = row.get("source_year") if hasattr(row, "get") else None
        try:
            source_year = int(source_year) if pd.notna(source_year) and str(source_year).strip() != "" else None
        except ValueError:
            source_year = None
        mapped["source_year_int"] = source_year
        mapped["time_heldout_2025plus"] = bool(mapped["eligible_for_frozen_scoring"] and source_year is not None and source_year >= 2025)
        mapped["time_heldout_2026only"] = bool(mapped["eligible_for_frozen_scoring"] and source_year == 2026)
        source_unit = row.get("source_name") if hasattr(row, "get") else None
        if pd.isna(source_unit) or str(source_unit).strip() == "":
            source_unit = row.get("source_id") if hasattr(row, "get") else None
        mapped["source_unit"] = None if pd.isna(source_unit) else str(source_unit).strip()
        rows.append(mapped)
    return pd.DataFrame(rows), pd.DataFrame(suggestions).drop_duplicates()


def add_subset_flags(prepared_df, benchmark_source_terms=None):
    benchmark_source_terms = [normalize_text(term) for term in (benchmark_source_terms or []) if term]
    df = prepared_df.copy()
    df["source_type_norm"] = df.get("source_type", pd.Series([None] * len(df))).map(normalize_text)
    df["source_name_norm"] = df.get("source_name", pd.Series([None] * len(df))).map(normalize_text)
    df["source_id_norm"] = df.get("source_id", pd.Series([None] * len(df))).map(normalize_text)

    def unseen_source(row):
        if not row["eligible_for_frozen_scoring"]:
            return False
        source_name = row.get("source_name_norm", "") or ""
        source_id = row.get("source_id_norm", "") or ""
        if benchmark_source_terms and any(term in source_name for term in benchmark_source_terms):
            return False
        if benchmark_source_terms and any(term in source_id for term in benchmark_source_terms):
            return False
        return True

    df["unseen_source"] = df.apply(unseen_source, axis=1)
    return df.drop(columns=["source_type_norm", "source_name_norm", "source_id_norm"])


def build_report(prepared_by_dataset):
    report = {"total_rows": int(len(next(iter(prepared_by_dataset.values()))))}
    for dataset, df in prepared_by_dataset.items():
        report[dataset] = {
            "fully_mapped": int(df["fully_mapped"].sum()),
            "already_in_benchmark": int(df["already_in_benchmark"].sum()),
            "eligible_for_frozen_scoring": int(df["eligible_for_frozen_scoring"].sum()),
            "resistance_rows": int((df["label_id"] == 1).sum()),
            "sensitivity_rows": int((df["label_id"] == 2).sum()),
            "time_heldout_2025plus": int(df["time_heldout_2025plus"].sum()) if "time_heldout_2025plus" in df else 0,
            "time_heldout_2026only": int(df["time_heldout_2026only"].sum()) if "time_heldout_2026only" in df else 0,
            "unseen_source": int(df["unseen_source"].sum()) if "unseen_source" in df else 0,
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare external ncRNA-drug-disease cases for frozen-checkpoint validation.")
    parser.add_argument("--external-csv", required=True)
    parser.add_argument("--dataset1-assoc", required=True)
    parser.add_argument("--dataset2-assoc", required=True)
    parser.add_argument("--dataset1-mapping", required=True)
    parser.add_argument("--dataset2-mapping", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--benchmark-source", action="append", default=[], help="Optional benchmark source keywords used to define unseen_source.")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    external_df = pd.read_csv(args.external_csv)
    missing_cols = [col for col in REQUIRED_EXTERNAL_COLUMNS if col not in external_df.columns]
    if missing_cols:
        raise ValueError(f"External CSV is missing required columns: {missing_cols}")
    external_df = external_df.copy()
    external_df["label_id"] = external_df["label"].map(normalize_label)

    configs = {
        "dataset1": {"assoc": args.dataset1_assoc, "mapping": args.dataset1_mapping},
        "dataset2": {"assoc": args.dataset2_assoc, "mapping": args.dataset2_mapping},
    }

    prepared_by_dataset = {}
    suggestion_frames = []
    for dataset, cfg in configs.items():
        mapping_df = load_mapping(cfg["mapping"])
        vocab, name_bank = build_vocab(mapping_df)
        known_triplets = load_known_triplets(cfg["assoc"], mapping_df)
        prepared_df, suggestions_df = map_external_rows(external_df, vocab, name_bank, known_triplets, dataset)
        prepared_df = add_subset_flags(prepared_df, args.benchmark_source)
        prepared_by_dataset[dataset] = prepared_df
        suggestion_frames.append(suggestions_df)
        prepared_df.to_csv(outdir / f"{dataset}_prepared.csv", index=False)

        eligible = prepared_df[prepared_df["eligible_for_frozen_scoring"]].copy()
        if not eligible.empty:
            eligible.to_csv(outdir / f"{dataset}_all_external_positive.csv", index=False)
            eligible[eligible["time_heldout_2025plus"]].to_csv(outdir / f"{dataset}_time_heldout_2025plus.csv", index=False)
            eligible[eligible["time_heldout_2026only"]].to_csv(outdir / f"{dataset}_time_heldout_2026only.csv", index=False)
            eligible[eligible["unseen_source"]].to_csv(outdir / f"{dataset}_unseen_source.csv", index=False)

    report = build_report(prepared_by_dataset)
    (outdir / "mapping_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if suggestion_frames:
        suggestions = pd.concat(suggestion_frames, ignore_index=True)
        if not suggestions.empty and "suggestions" in suggestions.columns:
            suggestions = suggestions[suggestions["suggestions"].fillna("") != ""]
            if not suggestions.empty:
                suggestions.to_csv(outdir / "unmapped_name_suggestions.csv", index=False)


if __name__ == "__main__":
    main()
