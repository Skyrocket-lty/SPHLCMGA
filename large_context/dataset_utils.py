from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _canonical_type(type_name: str) -> str:
    text = str(type_name).strip().lower()
    if text in {"ncrna", "ncrna ", "ncrna".lower()}:
        return "ncrna"
    if text in {"cancer", "disease"}:
        return "disease"
    return text


def load_mapping_table(path: Path, source_name: str) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df["type"] = df["type"].map(_canonical_type)
    mapping: dict[str, dict[int, str]] = {}
    for entity_type in ("ncrna", "drug", "disease"):
        sub = df[df["type"] == entity_type][["id", "name"]].copy()
        mapping[entity_type] = dict(zip(sub["id"].astype(int), sub["name"].astype(str)))
    df["source_dataset"] = source_name
    return df, mapping
