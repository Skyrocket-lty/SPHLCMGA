from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch

matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

from train_sphlcMga_core_vs_pathway import (
    ROOT,
    build_combined_features,
    build_relational_hypergraph,
    instantiate_model,
    load_aligned_context_aux_features,
)


LABEL_NAMES = {
    0: "non-association",
    1: "resistance",
    2: "sensitivity",
}


@dataclass
class RuntimeArtifacts:
    data_root: Path
    context_aux_root: Path
    rich_dir: Path
    differential_file: Path
    output_dir: Path
    external_validation_file: Path


def load_with_fold(result_dir: Path) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(result_dir.glob("fold_*")):
        frame = pd.read_csv(fold_dir / "validation_predictions.csv")
        frame["fold"] = fold_dir.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No fold validation predictions found under {result_dir}")
    return pd.concat(frames, ignore_index=True)


def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def load_nodes(data_root: Path) -> dict[str, pd.DataFrame]:
    node_dir = data_root / "nodes"
    return {
        key: pd.read_csv(node_dir / f"{key}.csv")
        for key in ["ncrna", "drug", "disease", "pathway"]
    }


def load_edges(data_root: Path) -> dict[str, pd.DataFrame]:
    edge_dir = data_root / "edges"
    return {
        key: pd.read_csv(edge_dir / f"{key}.csv")
        for key in ["rna_pathway", "drug_pathway", "disease_pathway"]
    }


def build_relation_tables(edges: dict[str, pd.DataFrame]) -> dict[str, dict[int, pd.DataFrame]]:
    return {
        "rna_pathway": {
            int(node_id): frame[["pathway_id", "weight"]].rename(columns={"weight": "rna_weight"})
            for node_id, frame in edges["rna_pathway"].groupby("ncrna_id")
        },
        "drug_pathway": {
            int(node_id): frame[["pathway_id", "weight"]].rename(columns={"weight": "drug_weight"})
            for node_id, frame in edges["drug_pathway"].groupby("drug_id")
        },
        "disease_pathway": {
            int(node_id): frame[["pathway_id", "weight"]].rename(columns={"weight": "disease_weight"})
            for node_id, frame in edges["disease_pathway"].groupby("disease_id")
        },
    }


def external_support_lookup(external_file: Path) -> set[tuple[str, str, str]]:
    if not external_file.exists():
        return set()
    external_df = pd.read_csv(external_file)
    if "eligible_for_frozen_scoring" in external_df.columns:
        external_df = external_df[external_df["eligible_for_frozen_scoring"] == True].copy()
    return set(
        zip(
            normalize_key(external_df["ncrna_canonical_name"]),
            normalize_key(external_df["drug_canonical_name"]),
            normalize_key(external_df["disease_canonical_name"]),
        )
    )


def compute_pathway_support(
    case_row: pd.Series,
    relation_tables: dict[str, dict[int, pd.DataFrame]],
    pathway_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ncrna_table = relation_tables["rna_pathway"].get(int(case_row["ncrna_id"]), pd.DataFrame(columns=["pathway_id", "rna_weight"]))
    drug_table = relation_tables["drug_pathway"].get(int(case_row["drug_id"]), pd.DataFrame(columns=["pathway_id", "drug_weight"]))
    disease_table = relation_tables["disease_pathway"].get(int(case_row["disease_id"]), pd.DataFrame(columns=["pathway_id", "disease_weight"]))

    shared = ncrna_table.merge(drug_table, on="pathway_id", how="inner").merge(disease_table, on="pathway_id", how="inner")
    shared["support_type"] = "shared_3way"
    if not shared.empty:
        shared["support_score"] = shared["rna_weight"] + shared["drug_weight"] + shared["disease_weight"]
        shared["entity_mask"] = "ncrna|drug|disease"

    pairwise_specs = [
        ("rna_disease", ncrna_table, disease_table, ["rna_weight", "disease_weight"], "ncrna|disease"),
        ("drug_disease", drug_table, disease_table, ["drug_weight", "disease_weight"], "drug|disease"),
        ("rna_drug", ncrna_table, drug_table, ["rna_weight", "drug_weight"], "ncrna|drug"),
    ]
    pairwise_frames = []
    pairwise_counts = {"rna_disease_count": 0, "drug_disease_count": 0, "rna_drug_count": 0}
    for support_type, left, right, columns, entity_mask in pairwise_specs:
        merged = left.merge(right, on="pathway_id", how="inner")
        pairwise_counts[f"{support_type}_count"] = int(len(merged))
        if merged.empty:
            continue
        merged["support_type"] = support_type
        merged["support_score"] = merged[columns].sum(axis=1)
        merged["entity_mask"] = entity_mask
        pairwise_frames.append(merged)

    if not shared.empty:
        motif_df = shared.sort_values("support_score", ascending=False).head(3).copy()
    elif pairwise_frames:
        pairwise_all = pd.concat(pairwise_frames, ignore_index=True)
        pairwise_all = pairwise_all.sort_values("support_score", ascending=False)
        motif_df = pairwise_all.drop_duplicates(subset=["pathway_id"]).head(3).copy()
    else:
        motif_df = pd.DataFrame(columns=["pathway_id", "support_type", "support_score", "entity_mask"])

    motif_df = motif_df.merge(
        pathway_nodes[["id", "name"]],
        left_on="pathway_id",
        right_on="id",
        how="left",
    ).drop(columns=["id"], errors="ignore")
    motif_df["pathway_name"] = motif_df["name"].fillna(motif_df["pathway_id"].astype(str))
    summary = {
        "shared_count": int(len(shared)),
        "shared_score": float(shared["support_score"].sum()) if not shared.empty else 0.0,
        "rna_disease_count": pairwise_counts["rna_disease_count"],
        "drug_disease_count": pairwise_counts["drug_disease_count"],
        "rna_drug_count": pairwise_counts["rna_drug_count"],
    }
    summary["viable_support"] = bool(
        summary["shared_count"] > 0
        or (summary["rna_disease_count"] > 0 and summary["drug_disease_count"] > 0)
        or (summary["rna_drug_count"] > 0 and summary["drug_disease_count"] > 0)
    )
    return motif_df, summary


def candidate_score(case_row: pd.Series, support_summary: dict[str, float]) -> float:
    confidence = float(case_row.get("confidence_rich", case_row.get("confidence", 0.0)))
    return (
        confidence
        + 0.03 * support_summary["shared_count"]
        + 0.004 * support_summary["shared_score"]
        + 0.008 * support_summary["drug_disease_count"]
        + 0.006 * support_summary["rna_disease_count"]
        + 0.004 * support_summary["rna_drug_count"]
    )


def prepare_case_pool(
    aligned_df: pd.DataFrame,
    rich_df: pd.DataFrame,
    nodes: dict[str, pd.DataFrame],
    relation_tables: dict[str, dict[int, pd.DataFrame]],
    nonzero_context_ids: set[int],
    external_hits: set[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ncrna_nodes = nodes["ncrna"][["id", "name", "canonical_name"]]
    drug_nodes = nodes["drug"][["id", "name", "canonical_name"]]
    disease_nodes = nodes["disease"][["id", "name", "canonical_name"]]

    def enrich(frame: pd.DataFrame, category: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        enriched = frame.copy()
        enriched = enriched[enriched["disease_id"].isin(nonzero_context_ids)].copy()
        if enriched.empty:
            return enriched
        enriched = (
            enriched
            .merge(ncrna_nodes, left_on="ncrna_id", right_on="id", how="left")
            .rename(columns={"name": "ncrna_name", "canonical_name": "ncrna_canonical"})
            .drop(columns=["id"])
            .merge(drug_nodes, left_on="drug_id", right_on="id", how="left")
            .rename(columns={"name": "drug_name", "canonical_name": "drug_canonical"})
            .drop(columns=["id"])
            .merge(disease_nodes, left_on="disease_id", right_on="id", how="left")
            .rename(columns={"name": "disease_name", "canonical_name": "disease_canonical"})
            .drop(columns=["id"])
        )
        records = []
        for row in enriched.itertuples(index=False):
            motif_df, support_summary = compute_pathway_support(pd.Series(row._asdict()), relation_tables, nodes["pathway"])
            record = row._asdict()
            record.update(support_summary)
            record["candidate_category"] = category
            record["support_score_total"] = candidate_score(pd.Series(record), support_summary)
            record["external_supported"] = (
                normalize_key(pd.Series([record["ncrna_canonical"]])).iloc[0],
                normalize_key(pd.Series([record["drug_canonical"]])).iloc[0],
                normalize_key(pd.Series([record["disease_canonical"]])).iloc[0],
            ) in external_hits
            record["top_pathway_names"] = "; ".join(motif_df["pathway_name"].head(3).tolist())
            records.append(record)
        return pd.DataFrame(records)

    corrected_resistance = enrich(
        aligned_df[
            (aligned_df["transition"] == "wrong_to_right")
            & (aligned_df["label"] == 1)
            & (aligned_df["pred_label_rich"] == 1)
        ],
        "corrected_resistance",
    )
    corrected_sensitivity = enrich(
        aligned_df[
            (aligned_df["transition"] == "wrong_to_right")
            & (aligned_df["label"] == 2)
            & (aligned_df["pred_label_rich"] == 2)
        ],
        "corrected_sensitivity",
    )
    novel_resistance = enrich(
        rich_df[
            (rich_df["label"] == 0)
            & (rich_df["pred_label"] == 1)
        ],
        "novel_resistance_candidate",
    )
    return corrected_resistance, corrected_sensitivity, novel_resistance


def select_case_rows(
    corrected_resistance: pd.DataFrame,
    corrected_sensitivity: pd.DataFrame,
    novel_resistance: pd.DataFrame,
    preferred_case: tuple[int, int, int] | None = None,
) -> tuple[pd.Series, list[pd.Series]]:
    def best_supported(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        supported = frame[frame["viable_support"] == True].copy()
        if supported.empty:
            supported = frame.copy()
        sort_keys = ["support_score_total", "shared_count"]
        if "confidence_rich" in supported.columns:
            sort_keys.append("confidence_rich")
        elif "confidence" in supported.columns:
            sort_keys.append("confidence")
        return supported.sort_values(sort_keys, ascending=False)

    corrected_resistance = best_supported(corrected_resistance)
    corrected_sensitivity = best_supported(corrected_sensitivity)
    novel_resistance = best_supported(novel_resistance)

    if corrected_resistance.empty:
        raise RuntimeError("No corrected resistance case with non-zero context features is available.")

    if preferred_case is not None:
        ncrna_id, drug_id, disease_id = preferred_case
        main_frame = pd.concat([corrected_resistance, corrected_sensitivity, novel_resistance], ignore_index=True)
        matched = main_frame[
            (main_frame["ncrna_id"] == int(ncrna_id))
            & (main_frame["drug_id"] == int(drug_id))
            & (main_frame["disease_id"] == int(disease_id))
        ]
        if matched.empty:
            raise ValueError(f"Preferred case {(ncrna_id, drug_id, disease_id)} was not found in the traced candidate pools.")
        main_case = matched.iloc[0]
    else:
        main_case = corrected_resistance.iloc[0]
    chosen = [main_case]
    for frame in [novel_resistance, corrected_sensitivity]:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            duplicate = any(
                int(prev["ncrna_id"]) == int(row["ncrna_id"])
                and int(prev["drug_id"]) == int(row["drug_id"])
                and int(prev["disease_id"]) == int(row["disease_id"])
                for prev in chosen
            )
            if not duplicate:
                chosen.append(row)
                break
        if len(chosen) >= 3:
            break

    if len(chosen) < 3:
        fallback = pd.concat([corrected_resistance, novel_resistance, corrected_sensitivity], ignore_index=True)
        confidence_key = "confidence_rich" if "confidence_rich" in fallback.columns else "confidence"
        fallback = fallback.sort_values(["support_score_total", confidence_key], ascending=False)
        for _, row in fallback.iterrows():
            duplicate = any(
                int(prev["ncrna_id"]) == int(row["ncrna_id"])
                and int(prev["drug_id"]) == int(row["drug_id"])
                and int(prev["disease_id"]) == int(row["disease_id"])
                for prev in chosen
            )
            if not duplicate:
                chosen.append(row)
            if len(chosen) >= 3:
                break

    return main_case, chosen[:3]


def build_context_support(
    case_row: pd.Series,
    artifacts: RuntimeArtifacts,
    rich_metrics: dict,
) -> pd.DataFrame:
    model_type = "context_pathway"
    fold_dir = artifacts.rich_dir / str(case_row["fold"])
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found for main case: {fold_dir}")

    device = torch.device("cpu")
    ncrna_feat, drug_feat, disease_feat = build_combined_features(artifacts.data_root)
    disease_aux_features, feature_columns, _ = load_aligned_context_aux_features(
        data_root=artifacts.data_root,
        context_aux_root=artifacts.context_aux_root,
        feature_groups=list(rich_metrics["context_feature_groups"]),
    )
    train_df = pd.read_csv(fold_dir / "train_split.csv")
    hyperedge_index, sign_weights = build_relational_hypergraph(
        train_df,
        n_ncrna=ncrna_feat.shape[0],
        n_drug=drug_feat.shape[0],
        device=device,
    )
    model, _, _ = instantiate_model(
        model_type=model_type,
        data_root=artifacts.data_root,
        device=device,
        ncrna_feat=ncrna_feat.to(device),
        drug_feat=drug_feat.to(device),
        disease_feat=disease_feat.to(device),
        relation_mode="all",
        disease_aux_features=disease_aux_features.to(device),
    )
    model.load_state_dict(torch.load(fold_dir / "context_pathway_model.pt", map_location=device))
    model.eval()

    ncrna_ids = torch.tensor([int(case_row["ncrna_id"])], dtype=torch.long, device=device)
    drug_ids = torch.tensor([int(case_row["drug_id"])], dtype=torch.long, device=device)
    disease_ids = torch.tensor([int(case_row["disease_id"])], dtype=torch.long, device=device)
    target_label = int(case_row.get("pred_label_rich", case_row.get("pred_label", case_row["label"])))

    with torch.no_grad():
        _, pred_probs, _ = model(
            ncrna_feat.to(device),
            drug_feat.to(device),
            disease_feat.to(device),
            ncrna_ids,
            drug_ids,
            disease_ids,
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights,
        )
        clean_probs = pred_probs[0].cpu().numpy()
        static_state = model.compute_triplet_static_state(
            ncrna_feat.to(device),
            drug_feat.to(device),
            disease_feat.to(device),
            ncrna_ids,
            drug_ids,
            disease_ids,
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights,
        )

    context_rows = []
    for feature_index, feature_name in enumerate(feature_columns):
        disease_override = disease_aux_features.clone().to(device)
        disease_override[int(case_row["disease_id"]), feature_index] = 0.0
        with torch.no_grad():
            logits = model.predict_with_context_aux_override(
                ncrna_feat.to(device),
                drug_feat.to(device),
                disease_feat.to(device),
                ncrna_ids,
                drug_ids,
                disease_ids,
                disease_aux_override=disease_override[disease_ids],
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights,
                static_state=static_state,
            )
            perturbed_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        context_rows.append(
            {
                "feature_name": feature_name,
                "clean_target_prob": float(clean_probs[target_label]),
                "perturbed_target_prob": float(perturbed_probs[target_label]),
                "support_delta": float(clean_probs[target_label] - perturbed_probs[target_label]),
                "feature_value": float(disease_aux_features[int(case_row["disease_id"]), feature_index].item()),
                "target_label": LABEL_NAMES[target_label],
            }
        )
    return pd.DataFrame(context_rows).sort_values(["support_delta", "feature_value"], ascending=False).reset_index(drop=True)


def render_case_figure(
    case_row: pd.Series,
    motif_df: pd.DataFrame,
    context_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(8.4, 4.9))
    fig.patch.set_facecolor("white")
    ax_graph = fig.add_axes([0.035, 0.09, 0.93, 0.83])
    ax_graph.set_facecolor("white")
    ax_graph.set_xlim(0, 1)
    ax_graph.set_ylim(0, 1)
    ax_graph.axis("off")

    style = {
        "ink": "#243447",
        "muted": "#64758B",
    }
    entity_colors = {
        "ncrna": "#5B84C4",
        "drug": "#E08A6A",
        "disease": "#C96A86",
        "pathway": "#5A9B90",
    }
    entity_fills = {
        "ncrna": "#EEF4FC",
        "drug": "#FCF0EA",
        "disease": "#FAEEF3",
        "pathway": "#EDF7F4",
    }

    def pretty_label(text: str) -> str:
        alias = {
            "lung non-small cell carcinoma": "NSCLC",
            "small cell lung cancer": "SCLC",
            "lung adenocarcinoma": "LUAD",
            "pancreatic cancer": "Pancreatic\ncancer",
            "cisplatin resistance lung signature": "Cisplatin resistance\nlung signature",
            "signaling by tgfb family members": "TGFB family\nsignaling",
            "signaling by tgf-beta receptor complex": "TGF-beta receptor\ncomplex",
            "tgf-beta receptor signaling activates smads": "TGF-beta activates\nSMADs",
        }
        key = str(text).strip().lower()
        if "tgf" in key and "receptor complex" in key:
            return "TGF-beta receptor\ncomplex"
        if "tgf" in key and "activates smads" in key:
            return "TGF-beta activates\nSMADs"
        if key in alias:
            return alias[key]
        return textwrap.fill(str(text), width=24)

    def draw_node(
        ax,
        center: tuple[float, float],
        text: str,
        color: str,
        width: float,
        height: float,
        key: str,
        subtitle: str | None = None,
    ) -> None:
        patch = FancyBboxPatch(
            (center[0] - width / 2, center[1] - height / 2),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            linewidth=1.55,
            edgecolor=color,
            facecolor=entity_fills[key],
        )
        ax.add_patch(patch)
        title_y = center[1] + (0.035 if subtitle else 0.0)
        ax.text(center[0], title_y, text, ha="center", va="center", fontsize=9.6, color=style["ink"])
        if subtitle:
            ax.text(center[0], center[1] - 0.040, subtitle, ha="center", va="center", fontsize=8.3, color=style["muted"])

    def draw_edge(start: tuple[float, float], end: tuple[float, float], color: str, lw: float = 1.4) -> None:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            alpha=0.88,
            connectionstyle="arc3,rad=0.0",
            shrinkA=6,
            shrinkB=6,
        )
        ax_graph.add_patch(arrow)

    entity_pos = {
        "ncrna": (0.14, 0.72),
        "drug": (0.14, 0.28),
        "disease": (0.87, 0.50),
    }
    pathway_y = [0.76, 0.50, 0.24]
    pathway_pos = [(0.50, y) for y in pathway_y[: max(len(motif_df), 1)]]

    draw_node(ax_graph, entity_pos["ncrna"], pretty_label(str(case_row["ncrna_name"])), entity_colors["ncrna"], 0.33, 0.14, "ncrna")
    draw_node(ax_graph, entity_pos["drug"], pretty_label(str(case_row["drug_name"])), entity_colors["drug"], 0.23, 0.12, "drug")
    draw_node(ax_graph, entity_pos["disease"], pretty_label(str(case_row["disease_name"])), entity_colors["disease"], 0.20, 0.12, "disease")

    for idx, motif in enumerate(motif_df.itertuples(index=False)):
        node_pos = pathway_pos[idx]
        draw_node(
            ax_graph,
            node_pos,
            pretty_label(str(motif.pathway_name)),
            entity_colors["pathway"],
            0.38,
            0.22,
            "pathway",
            subtitle=f"support = {float(motif.support_score):.2f}",
        )
        mask = str(motif.entity_mask)
        if "ncrna" in mask:
            draw_edge((entity_pos["ncrna"][0] + 0.12, entity_pos["ncrna"][1]), (node_pos[0] - 0.18, node_pos[1]), entity_colors["ncrna"], 1.35)
        if "drug" in mask:
            draw_edge((entity_pos["drug"][0] + 0.10, entity_pos["drug"][1]), (node_pos[0] - 0.18, node_pos[1]), entity_colors["drug"], 1.35)
        if "disease" in mask:
            disease_targets = [(entity_pos["disease"][0] - 0.10, 0.56), (entity_pos["disease"][0] - 0.10, 0.50), (entity_pos["disease"][0] - 0.10, 0.44)]
            draw_edge((node_pos[0] + 0.18, node_pos[1]), disease_targets[min(idx, len(disease_targets) - 1)], entity_colors["disease"], 1.45)

    ax_graph.text(0.00, 1.02, "Pathway-context support motif", transform=ax_graph.transAxes, fontsize=12.0, weight="bold", color=style["ink"], va="bottom")
    subtitle = (
        f"{case_row['ncrna_name']} -- {case_row['drug_name']} -- NSCLC\n"
        f"Standard: {LABEL_NAMES[int(case_row['pred_label_standard'])]} ({float(case_row['confidence_standard']):.3f}) | "
        f"Rich: {LABEL_NAMES[int(case_row['pred_label_rich'])]} ({float(case_row['confidence_rich']):.3f})"
    )
    ax_graph.text(0.00, 0.97, subtitle, transform=ax_graph.transAxes, fontsize=8.9, color=style["muted"], va="top")

    fig.savefig(output_dir / "evidence_trace_case_graph.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "evidence_trace_case_graph.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace pathway/context support motifs for the final SPHLCMGA-CP model.")
    parser.add_argument("--rich-dir", type=str, default=str(ROOT / "outputs_sphlcMga_pathway_context_study" / "context_pathway_all"))
    parser.add_argument("--differential-file", type=str, default=str(ROOT / "outputs_sphlcMga_pathway_context_study" / "differential_gain" / "aligned_oof_predictions.csv"))
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data" / "processed_pathway_only"))
    parser.add_argument("--context-aux-root", type=str, default=str(ROOT / "data" / "processed"))
    parser.add_argument("--external-validation-file", type=str, default=str(ROOT.parent / "external_validation" / "prepared_outputs_example" / "dataset2_all_external_positive.csv"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs_sphlcMga_pathway_context_study" / "evidence_trace"))
    parser.add_argument("--preferred-ncrna-id", type=int, default=1002)
    parser.add_argument("--preferred-drug-id", type=int, default=21)
    parser.add_argument("--preferred-disease-id", type=int, default=17)
    args = parser.parse_args()

    artifacts = RuntimeArtifacts(
        data_root=Path(args.data_root),
        context_aux_root=Path(args.context_aux_root),
        rich_dir=Path(args.rich_dir),
        differential_file=Path(args.differential_file),
        output_dir=Path(args.output_dir),
        external_validation_file=Path(args.external_validation_file),
    )
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    rich_metrics = json.loads((artifacts.rich_dir / "metrics.json").read_text(encoding="utf-8"))
    rich_df = load_with_fold(artifacts.rich_dir)
    aligned_df = pd.read_csv(artifacts.differential_file)
    nodes = load_nodes(artifacts.data_root)
    edges = load_edges(artifacts.data_root)
    relation_tables = build_relation_tables(edges)
    disease_aux, feature_columns, _ = load_aligned_context_aux_features(
        data_root=artifacts.data_root,
        context_aux_root=artifacts.context_aux_root,
        feature_groups=list(rich_metrics["context_feature_groups"]),
    )
    nonzero_context_ids = set(np.where(np.abs(disease_aux.numpy()).sum(axis=1) > 0)[0].astype(int).tolist())
    external_hits = external_support_lookup(artifacts.external_validation_file)

    corrected_resistance, corrected_sensitivity, novel_resistance = prepare_case_pool(
        aligned_df=aligned_df,
        rich_df=rich_df,
        nodes=nodes,
        relation_tables=relation_tables,
        nonzero_context_ids=nonzero_context_ids,
        external_hits=external_hits,
    )
    preferred_case = None
    if args.preferred_ncrna_id >= 0 and args.preferred_drug_id >= 0 and args.preferred_disease_id >= 0:
        preferred_case = (args.preferred_ncrna_id, args.preferred_drug_id, args.preferred_disease_id)
    main_case, top_cases = select_case_rows(corrected_resistance, corrected_sensitivity, novel_resistance, preferred_case=preferred_case)

    motif_df, support_summary = compute_pathway_support(main_case, relation_tables, nodes["pathway"])
    context_df = build_context_support(main_case, artifacts, rich_metrics)
    supportive_context = context_df[context_df["support_delta"] > 0].copy()
    if supportive_context.empty:
        supportive_context = context_df.head(3).copy()
    else:
        supportive_context = supportive_context.head(3).copy()

    topk_rows = []
    for rank, case in enumerate(top_cases, start=1):
        case_motifs, _ = compute_pathway_support(case, relation_tables, nodes["pathway"])
        case_context_df = build_context_support(case, artifacts, rich_metrics)
        case_supportive_context = case_context_df[case_context_df["support_delta"] > 0].copy()
        if case_supportive_context.empty:
            case_supportive_context = case_context_df.head(3).copy()
        else:
            case_supportive_context = case_supportive_context.head(3).copy()
        topk_rows.append(
            {
                "rank": rank,
                "case_category": case["candidate_category"],
                "fold": case["fold"],
                "triplet": f"{case['ncrna_name']} -- {case['drug_name']} -- {case['disease_name']}",
                "benchmark_status": "known_positive" if int(case["label"]) in {1, 2} else "candidate_negative_pool",
                "target_label": LABEL_NAMES[int(case["label"])],
                "rich_prediction": LABEL_NAMES[int(case.get("pred_label_rich", case.get("pred_label", 0)))],
                "rich_confidence": float(case.get("confidence_rich", case.get("confidence", 0.0))),
                "top_pathways": "; ".join(case_motifs["pathway_name"].head(3).tolist()),
                "top_context_features": "; ".join(case_supportive_context["feature_name"].head(3).tolist()),
                "external_supported": bool(case["external_supported"]),
                "support_summary": json.dumps(
                    {
                        "shared_count": int(case["shared_count"]),
                        "shared_score": float(case["shared_score"]),
                        "rna_disease_count": int(case["rna_disease_count"]),
                        "drug_disease_count": int(case["drug_disease_count"]),
                        "rna_drug_count": int(case["rna_drug_count"]),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    case_row = {
        "case_id": "pathway_context_main_case",
        "case_category": str(main_case["candidate_category"]),
        "fold": str(main_case["fold"]),
        "triplet": f"{main_case['ncrna_name']} -- {main_case['drug_name']} -- {main_case['disease_name']}",
        "ncrna_id": int(main_case["ncrna_id"]),
        "drug_id": int(main_case["drug_id"]),
        "disease_id": int(main_case["disease_id"]),
        "ncrna_name": str(main_case["ncrna_name"]),
        "drug_name": str(main_case["drug_name"]),
        "disease_name": str(main_case["disease_name"]),
        "label": int(main_case["label"]),
        "label_name": LABEL_NAMES[int(main_case["label"])],
        "standard_prediction": LABEL_NAMES[int(main_case["pred_label_standard"])],
        "rich_prediction": LABEL_NAMES[int(main_case["pred_label_rich"])],
        "standard_confidence": float(main_case["confidence_standard"]),
        "rich_confidence": float(main_case["confidence_rich"]),
        "standard_resistance_prob": float(main_case["prob_resistance_standard"]),
        "standard_sensitivity_prob": float(main_case["prob_sensitivity_standard"]),
        "rich_resistance_prob": float(main_case["prob_resistance_rich"]),
        "rich_sensitivity_prob": float(main_case["prob_sensitivity_rich"]),
        "transition": str(main_case["transition"]),
        "support_mode": "shared_3way" if support_summary["shared_count"] > 0 else "pairwise_convergent",
        "selected_pathway_ids": ";".join(str(int(item)) for item in motif_df["pathway_id"].head(3).tolist()),
        "selected_pathway_names": "; ".join(motif_df["pathway_name"].head(3).tolist()),
        "selected_context_features": "; ".join(supportive_context["feature_name"].head(3).tolist()),
        "external_supported": bool(main_case["external_supported"]),
        "selection_reason": "corrected polarity-sensitive resistance case with exact pathway convergence and non-zero context cues",
    }

    pd.DataFrame([case_row]).to_csv(artifacts.output_dir / "evidence_trace_case.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(artifacts.output_dir / "evidence_trace_topk.csv", index=False)
    motif_df.to_csv(artifacts.output_dir / "evidence_trace_case_motifs.csv", index=False)
    context_df.to_csv(artifacts.output_dir / "evidence_trace_case_context.csv", index=False)
    render_case_figure(main_case, motif_df.head(3), supportive_context.head(3), artifacts.output_dir)

    metadata = {
        "rich_dir": str(artifacts.rich_dir),
        "differential_file": str(artifacts.differential_file),
        "data_root": str(artifacts.data_root),
        "context_aux_root": str(artifacts.context_aux_root),
        "context_feature_columns": feature_columns,
        "selected_case": case_row,
        "topk_count": int(len(topk_rows)),
    }
    (artifacts.output_dir / "evidence_trace_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata["selected_case"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


