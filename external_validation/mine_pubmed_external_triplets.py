import argparse
import csv
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests


DEFAULT_QUERY = (
    "((2024:2025[pdat]) AND (lncrna OR microrna OR mirna OR circrna) "
    "AND (resistance OR sensitivity OR chemosensitivity OR sensitizes OR sensitization OR restores sensitivity) "
    "AND (cisplatin OR doxorubicin OR paclitaxel OR fluorouracil OR gemcitabine OR oxaliplatin OR "
    "tamoxifen OR sorafenib OR gefitinib OR docetaxel OR temozolomide))"
)


def normalize_sentence(text):
    return (
        str(text)
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .strip()
    )


def split_sentences(text):
    text = normalize_sentence(text)
    return [part.strip() for part in re.split(r"(?<=[\.!?])\s+", text) if part.strip()]


def build_name_patterns(mapping_paths):
    frames = [pd.read_excel(path) for path in mapping_paths]
    mapping_df = pd.concat(frames, ignore_index=True).drop_duplicates(["type", "name"])

    def compile_pattern(name):
        text = normalize_sentence(name)
        parts = [re.escape(part) for part in re.split(r"\s+", text) if part]
        body = r"\s+".join(parts)
        return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.I)

    entity_lists = {
        "ncrna": sorted(set(mapping_df[mapping_df["type"] == "ncRNA"]["name"].astype(str)), key=len, reverse=True),
        "drug": sorted(set(mapping_df[mapping_df["type"] == "drug"]["name"].astype(str)), key=len, reverse=True),
        "disease": sorted(set(mapping_df[mapping_df["type"].isin(["cancer", "disease"])]["name"].astype(str)), key=len, reverse=True),
    }
    return {
        entity_type: [(name, compile_pattern(name)) for name in names]
        for entity_type, names in entity_lists.items()
    }


def infer_label(sentence):
    low = sentence.lower()
    if "retract" in low:
        return None

    nc_suppression_cues = [
        "silencing",
        "knockdown",
        "downregulation",
        "downregulated",
        "suppression",
        "suppresses",
        "suppressed",
        "inhibition of",
        "blocking",
        "blocks",
        "decreased expression",
        "si-",
    ]
    resistance_reversal_cues = [
        "restores sensitivity",
        "restore sensitivity",
        "chemosensitivity",
        "sensitization",
        "sensitize",
        "sensitizes",
        "reverses resistance",
        "reverse resistance",
        "reverses cisplatin resistance",
        "overcome resistance",
        "overcomes resistance",
    ]

    # If suppression of the ncRNA restores sensitivity or reverses resistance,
    # the ncRNA itself behaves as a resistance-associated factor.
    if any(token in low for token in nc_suppression_cues) and any(token in low for token in resistance_reversal_cues):
        return "resistance"

    # explicit sensitivity/restoration cues
    if any(token in low for token in ["restores sensitivity", "restore sensitivity", "chemosensitivity", "sensitization", "sensitize", "sensitizes"]):
        return "sensitivity"
    if any(token in low for token in ["enhances sensitivity", "increase sensitivity", "improves sensitivity", "facilitates chemosensitivity", "reverses cisplatin resistance", "reverses resistance", "reverse resistance", "overcome cisplatin resistance", "overcomes resistance"]):
        return "sensitivity"

    # explicit reduced sensitivity cues imply resistance
    if any(token in low for token in ["decreases sensitivity", "decrease sensitivity", "reduced sensitivity", "reduces sensitivity", "loss of sensitivity", "attenuates sensitivity"]):
        return "resistance"

    if "resistan" in low and "sensitiv" not in low:
        return "resistance"
    if "sensitiv" in low and "resistan" not in low:
        return "sensitivity"
    return None


def match_entities(sentence, patterns):
    hits = {}
    for entity_type, compiled in patterns.items():
        names = [name for name, regex in compiled if regex.search(sentence)]
        hits[entity_type] = names
    return hits


def fetch_pubmed_records(query, retmax=400, pause_sec=0.34):
    search_resp = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"},
        timeout=60,
    )
    search_resp.raise_for_status()
    result = search_resp.json()["esearchresult"]
    ids = result["idlist"]
    records = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        fetch_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
            timeout=120,
        )
        fetch_resp.raise_for_status()
        if not fetch_resp.text.strip():
            continue
        root = ET.fromstring(fetch_resp.text)
        for article in root.findall('.//PubmedArticle'):
            pmid = article.findtext('.//PMID', '')
            title = ' '.join(article.findtext('.//ArticleTitle', '').split())
            abstract = ' '.join(' '.join(node.itertext()) for node in article.findall('.//AbstractText'))
            year = article.findtext('.//PubDate/Year', '') or article.findtext('.//PubMedPubDate[@PubStatus="pubmed"]/Year', '')
            month = article.findtext('.//PubDate/Month', '')
            day = article.findtext('.//PubDate/Day', '')
            records.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "source_year": year,
                    "source_date": "-".join(part for part in [year, month, day] if part),
                }
            )
        time.sleep(pause_sec)
    return records


def build_candidate_rows(records, patterns):
    rows = []
    seen = set()
    for record in records:
        text = " ".join([record["title"], record["abstract"]])
        for sentence in split_sentences(text):
            label = infer_label(sentence)
            if label is None:
                continue
            hits = match_entities(sentence, patterns)
            if not hits["ncrna"] or not hits["drug"] or not hits["disease"]:
                continue

            # Keep the longest matches first to reduce generic duplicates.
            rna_name = hits["ncrna"][0]
            drug_name = hits["drug"][0]
            disease_name = hits["disease"][0]
            key = (record["pmid"], rna_name, drug_name, disease_name, label)
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "ncrna_name": rna_name,
                    "drug_name": drug_name,
                    "disease_name": disease_name,
                    "label": label,
                    "source_year": record["source_year"],
                    "source_id": record["pmid"],
                    "source_name": f"PMID:{record['pmid']}",
                    "source_type": "publication",
                    "source_date": record["source_date"],
                    "disease_context_note": sentence,
                    "evidence_level": "sentence-level",
                    "evidence_note": record["title"],
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Mine 2024-2025 PubMed records for external ncRNA-drug-disease validation candidates.")
    parser.add_argument("--dataset1-mapping", required=True)
    parser.add_argument("--dataset2-mapping", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--retmax", type=int, default=400)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    return parser.parse_args()


def main():
    args = parse_args()
    patterns = build_name_patterns([Path(args.dataset1_mapping), Path(args.dataset2_mapping)])
    records = fetch_pubmed_records(args.query, retmax=args.retmax)
    rows = build_candidate_rows(records, patterns)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "ncrna_name",
            "drug_name",
            "disease_name",
            "label",
            "source_year",
            "source_id",
            "source_name",
            "source_type",
            "source_date",
            "disease_context_note",
            "evidence_level",
            "evidence_note",
        ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
