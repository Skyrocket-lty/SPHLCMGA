from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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


def clone_relations(relations: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        relation_name: {key: tensor.clone() for key, tensor in relation.items()}
        for relation_name, relation in relations.items()
    }


def filter_case_pathway_relations(
    relations: dict[str, dict[str, torch.Tensor]],
    *,
    ncrna_id: int,
    drug_id: int,
    disease_id: int,
    selected_pathway_ids: list[int],
) -> dict[str, dict[str, torch.Tensor]]:
    filtered = clone_relations(relations)
    for relation_name, left_prefix, target_id in [
        ("rna_pathway", "ncrna", ncrna_id),
        ("drug_pathway", "drug", drug_id),
        ("disease_pathway", "disease", disease_id),
    ]:
        left_key = f"{left_prefix}_index"
        source = filtered[relation_name][left_key]
        pathway = filtered[relation_name]["pathway_index"]
        weight = filtered[relation_name]["weight"]
        removal_set = torch.tensor(selected_pathway_ids, dtype=pathway.dtype, device=pathway.device)
        keep_mask = ~((source == int(target_id)) & torch.isin(pathway, removal_set))
        filtered[relation_name][left_key] = source[keep_mask]
        filtered[relation_name]["pathway_index"] = pathway[keep_mask]
        filtered[relation_name]["weight"] = weight[keep_mask]
    return filtered


def load_case(case_file: Path) -> pd.Series:
    case_df = pd.read_csv(case_file)
    if case_df.empty:
        raise ValueError(f"No rows found in case file: {case_file}")
    return case_df.iloc[0]


def load_main_model(
    checkpoint_dir: Path,
    data_root: Path,
    context_aux_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], pd.DataFrame]:
    root_metrics = json.loads((checkpoint_dir.parent / "metrics.json").read_text(encoding="utf-8"))
    context_feature_groups = list(root_metrics["context_feature_groups"])
    ncrna_feat, drug_feat, disease_feat = build_combined_features(data_root)
    disease_aux, feature_columns, _ = load_aligned_context_aux_features(
        data_root=data_root,
        context_aux_root=context_aux_root,
        feature_groups=context_feature_groups,
    )
    model, _, _ = instantiate_model(
        model_type="context_pathway",
        data_root=data_root,
        device=device,
        ncrna_feat=ncrna_feat.to(device),
        drug_feat=drug_feat.to(device),
        disease_feat=disease_feat.to(device),
        relation_mode="all",
        disease_aux_features=disease_aux.to(device),
    )
    model.load_state_dict(torch.load(checkpoint_dir / "context_pathway_model.pt", map_location=device))
    model.eval()
    train_df = pd.read_csv(checkpoint_dir / "train_split.csv")
    return model, ncrna_feat, drug_feat, disease_feat, disease_aux, feature_columns, train_df


def build_ncrna_dual_knockout(
    ncrna_feat: torch.Tensor,
    train_df: pd.DataFrame,
    *,
    ncrna_id: int,
    drug_feat: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    knocked_ncrna_feat = ncrna_feat.clone()
    knocked_ncrna_feat[ncrna_id] = 0.0
    filtered_train_df = train_df[train_df["ncrna_id"] != int(ncrna_id)].copy()
    hyperedge_index, sign_weights = build_relational_hypergraph(
        filtered_train_df,
        n_ncrna=ncrna_feat.shape[0],
        n_drug=drug_feat.shape[0],
        device=device,
    )
    return knocked_ncrna_feat, (hyperedge_index, sign_weights)


def predict_case(
    *,
    model: torch.nn.Module,
    ncrna_feat: torch.Tensor,
    drug_feat: torch.Tensor,
    disease_feat: torch.Tensor,
    disease_aux: torch.Tensor,
    ncrna_id: int,
    drug_id: int,
    disease_id: int,
    device: torch.device,
    hypergraph: tuple[torch.Tensor, torch.Tensor],
    relation_override: dict[str, dict[str, torch.Tensor]] | None = None,
    context_feature_indices_to_zero: list[int] | None = None,
) -> np.ndarray:
    old_relations = None
    if relation_override is not None:
        old_relations = model.relations
        model.relations = relation_override

    ncrna_ids = torch.tensor([int(ncrna_id)], dtype=torch.long, device=device)
    drug_ids = torch.tensor([int(drug_id)], dtype=torch.long, device=device)
    disease_ids = torch.tensor([int(disease_id)], dtype=torch.long, device=device)
    hyperedge_index, sign_weights = hypergraph

    with torch.no_grad():
        if context_feature_indices_to_zero:
            disease_override = disease_aux.clone().to(device)
            disease_override[int(disease_id), context_feature_indices_to_zero] = 0.0
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
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        else:
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
            probs = pred_probs[0].cpu().numpy()

    if old_relations is not None:
        model.relations = old_relations
    return probs


def render_probability_shift_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = {
        "prob_non_association": "#8FA0C8",
        "prob_resistance": "#D46A6A",
        "prob_sensitivity": "#7BB17C",
    }
    x = np.arange(len(summary_df))
    width = 0.24
    ax.bar(x - width, summary_df["prob_non_association"], width=width, color=colors["prob_non_association"], label="Non-association")
    ax.bar(x, summary_df["prob_resistance"], width=width, color=colors["prob_resistance"], label="Resistance")
    ax.bar(x + width, summary_df["prob_sensitivity"], width=width, color=colors["prob_sensitivity"], label="Sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["mode_label"], rotation=18, ha="right")
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(output_dir / "ko_probability_shift_plot.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ko_probability_shift_plot.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ncRNA/pathway/context knockout consistency analysis for the selected SPHLCMGA-CP case.")
    parser.add_argument("--checkpoint-root", type=str, default=str(ROOT / "outputs_sphlcMga_pathway_context_study" / "context_pathway_all"))
    parser.add_argument("--case-file", type=str, default=str(ROOT / "outputs_sphlcMga_pathway_context_study" / "evidence_trace" / "evidence_trace_case.csv"))
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data" / "processed_pathway_only"))
    parser.add_argument("--context-aux-root", type=str, default=str(ROOT / "data" / "processed"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs_sphlcMga_pathway_context_study" / "ko_consistency"))
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    case = load_case(Path(args.case_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    fold_dir = checkpoint_root / str(case["fold"])
    model, ncrna_feat, drug_feat, disease_feat, disease_aux, feature_columns, train_df = load_main_model(
        checkpoint_dir=fold_dir,
        data_root=Path(args.data_root),
        context_aux_root=Path(args.context_aux_root),
        device=device,
    )
    base_hypergraph = build_relational_hypergraph(
        train_df,
        n_ncrna=ncrna_feat.shape[0],
        n_drug=drug_feat.shape[0],
        device=device,
    )

    ncrna_id = int(case["ncrna_id"])
    drug_id = int(case["drug_id"])
    disease_id = int(case["disease_id"])
    target_label = int(case["label"])

    selected_pathway_ids = [int(item) for item in str(case["selected_pathway_ids"]).split(";") if str(item).strip()]
    primary_pathway_ids = selected_pathway_ids[:1]
    selected_context_features = [item.strip() for item in str(case["selected_context_features"]).split(";") if item.strip()]
    context_indices = [feature_columns.index(name) for name in selected_context_features if name in feature_columns]

    clean_probs = predict_case(
        model=model,
        ncrna_feat=ncrna_feat,
        drug_feat=drug_feat,
        disease_feat=disease_feat,
        disease_aux=disease_aux,
        ncrna_id=ncrna_id,
        drug_id=drug_id,
        disease_id=disease_id,
        device=device,
        hypergraph=base_hypergraph,
    )

    knocked_ncrna_feat, knocked_hypergraph = build_ncrna_dual_knockout(
        ncrna_feat=ncrna_feat,
        train_df=train_df,
        ncrna_id=ncrna_id,
        drug_feat=drug_feat,
        device=device,
    )
    pathway_relations_ko = filter_case_pathway_relations(
        model.relations,
        ncrna_id=ncrna_id,
        drug_id=drug_id,
        disease_id=disease_id,
        selected_pathway_ids=primary_pathway_ids,
    )

    mode_outputs = [
        ("clean", "Clean", clean_probs),
        ("ncrna_ko", "ncRNA KO", predict_case(model=model, ncrna_feat=knocked_ncrna_feat, drug_feat=drug_feat, disease_feat=disease_feat, disease_aux=disease_aux, ncrna_id=ncrna_id, drug_id=drug_id, disease_id=disease_id, device=device, hypergraph=knocked_hypergraph)),
        ("pathway_ko", "Pathway KO", predict_case(model=model, ncrna_feat=ncrna_feat, drug_feat=drug_feat, disease_feat=disease_feat, disease_aux=disease_aux, ncrna_id=ncrna_id, drug_id=drug_id, disease_id=disease_id, device=device, hypergraph=base_hypergraph, relation_override=pathway_relations_ko)),
        ("context_ko", "Context KO", predict_case(model=model, ncrna_feat=ncrna_feat, drug_feat=drug_feat, disease_feat=disease_feat, disease_aux=disease_aux, ncrna_id=ncrna_id, drug_id=drug_id, disease_id=disease_id, device=device, hypergraph=base_hypergraph, context_feature_indices_to_zero=context_indices)),
        ("ncrna_pathway_ko", "ncRNA + Pathway KO", predict_case(model=model, ncrna_feat=knocked_ncrna_feat, drug_feat=drug_feat, disease_feat=disease_feat, disease_aux=disease_aux, ncrna_id=ncrna_id, drug_id=drug_id, disease_id=disease_id, device=device, hypergraph=knocked_hypergraph, relation_override=pathway_relations_ko)),
        ("ncrna_context_ko", "ncRNA + Context KO", predict_case(model=model, ncrna_feat=knocked_ncrna_feat, drug_feat=drug_feat, disease_feat=disease_feat, disease_aux=disease_aux, ncrna_id=ncrna_id, drug_id=drug_id, disease_id=disease_id, device=device, hypergraph=knocked_hypergraph, context_feature_indices_to_zero=context_indices)),
    ]

    clean_target_prob = float(clean_probs[target_label])
    rows = []
    mode_lookup = {}
    for mode, mode_label, probs in mode_outputs:
        rows.append(
            {
                "mode": mode,
                "mode_label": mode_label,
                "predicted_class": LABEL_NAMES[int(np.argmax(probs))],
                "prob_non_association": float(probs[0]),
                "prob_resistance": float(probs[1]),
                "prob_sensitivity": float(probs[2]),
                f"delta_{LABEL_NAMES[target_label]}": float(clean_target_prob - probs[target_label]),
                "selected_pathway": "; ".join(str(item) for item in primary_pathway_ids) if "pathway" in mode else "",
                "selected_context_features": "; ".join(selected_context_features) if "context" in mode else "",
            }
        )
        mode_lookup[mode] = probs

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "ko_consistency_summary.csv", index=False)
    render_probability_shift_plot(summary_df, output_dir)

    def delta_target(probs: np.ndarray) -> float:
        return float(clean_target_prob - probs[target_label])

    metadata = {
        "target_label": LABEL_NAMES[target_label],
        "clean_target_prob": clean_target_prob,
        "pathway_synergy": float(
            delta_target(mode_lookup["ncrna_pathway_ko"])
            - delta_target(mode_lookup["ncrna_ko"])
            - delta_target(mode_lookup["pathway_ko"])
        ),
        "context_synergy": float(
            delta_target(mode_lookup["ncrna_context_ko"])
            - delta_target(mode_lookup["ncrna_ko"])
            - delta_target(mode_lookup["context_ko"])
        ),
        "selected_pathway_ids": primary_pathway_ids,
        "selected_context_features": selected_context_features,
        "case_triplet": f"{case['ncrna_name']} -- {case['drug_name']} -- {case['disease_name']}",
        "case_fold": str(case["fold"]),
    }
    (output_dir / "ko_consistency_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

