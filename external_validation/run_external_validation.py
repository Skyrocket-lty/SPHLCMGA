import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_CANDIDATES = [
    SCRIPT_PATH.parents[2],
    SCRIPT_PATH.parents[1],
    SCRIPT_PATH.parent,
]
for candidate in PROJECT_CANDIDATES:
    cold_start_dir = candidate / "cold_start_work"
    if cold_start_dir.exists() and str(cold_start_dir) not in sys.path:
        sys.path.insert(0, str(cold_start_dir))
    if (candidate / "dataset2_model.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dataset2_model import NcrnaDrugDiseaseDualBranchModel, parameters_set  # noqa: E402


LABEL_NAMES = {0: "non-association", 1: "resistance", 2: "sensitivity"}


def build_relational_hypergraph(train_data, ncrna_num, drug_num, disease_num, device):
    resistant_data = train_data[train_data[:, 3] == 1]
    sensitive_data = train_data[train_data[:, 3] == 2]

    def build_single(data, sign):
        if len(data) == 0:
            return np.array([[], []]), np.array([])
        ncrna_ids = data[:, 0].astype(int)
        drug_ids = data[:, 1].astype(int) + ncrna_num
        disease_ids = data[:, 2].astype(int) + ncrna_num + drug_num
        hyperedge_nodes = np.vstack([ncrna_ids, drug_ids, disease_ids]).T
        hyperedge_index = []
        for edge_idx, nodes in enumerate(hyperedge_nodes):
            for node in nodes:
                hyperedge_index.append([node, edge_idx])
        return np.array(hyperedge_index).T, np.full(len(hyperedge_nodes), sign, dtype=np.float32)

    res_edge, res_sign = build_single(resistant_data, -1.0)
    sen_edge, sen_sign = build_single(sensitive_data, +1.0)

    if res_edge.size == 0 and sen_edge.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float32, device=device)
    if res_edge.size == 0:
        return torch.tensor(sen_edge, dtype=torch.long, device=device), torch.tensor(sen_sign, dtype=torch.float32, device=device)
    if sen_edge.size == 0:
        return torch.tensor(res_edge, dtype=torch.long, device=device), torch.tensor(res_sign, dtype=torch.float32, device=device)

    res_edge[1, :] += len(sen_sign)
    hyperedge_index = np.hstack([sen_edge, res_edge])
    sign_weights = np.hstack([sen_sign, res_sign])
    return torch.tensor(hyperedge_index, dtype=torch.long, device=device), torch.tensor(sign_weights, dtype=torch.float32, device=device)


def load_feature_tensors(data_dir, device):
    rna_base_path = data_dir / "RNA.txt"
    if not rna_base_path.exists():
        rna_base_path = data_dir / "ncRNA.txt"
    ncrna_feat = (
        np.loadtxt(data_dir / "LLM_rna_sim.txt") + np.loadtxt(rna_base_path)
    ) / 2.0
    drug_feat = (
        np.loadtxt(data_dir / "LLM_drug_sim.txt") + np.loadtxt(data_dir / "drug.txt")
    ) / 2.0
    disease_feat = (
        np.loadtxt(data_dir / "LLM_disease_sim.txt") + np.loadtxt(data_dir / "disease.txt")
    ) / 2.0
    return (
        torch.from_numpy(ncrna_feat.astype(np.float32)).to(device),
        torch.from_numpy(drug_feat.astype(np.float32)).to(device),
        torch.from_numpy(disease_feat.astype(np.float32)).to(device),
    )


def load_model(checkpoint_path, dims, device):
    args = parameters_set()
    model = NcrnaDrugDiseaseDualBranchModel(
        ncrna_dim=dims[0],
        drug_dim=dims[1],
        disease_dim=dims[2],
        modal_hidden_dim=args.modal_hidden_dim,
        modal_out_dim=args.modal_out_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        hgnn_dim_1=args.hgnn_dim_1,
        dropout=args.dropout,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_triplets(model, ncrna_feat, drug_feat, disease_feat, triplets, hyperedge_index, sign_weights):
    with torch.no_grad():
        ncrna_ids = torch.tensor(triplets[:, 0], dtype=torch.long, device=ncrna_feat.device)
        drug_ids = torch.tensor(triplets[:, 1], dtype=torch.long, device=ncrna_feat.device)
        disease_ids = torch.tensor(triplets[:, 2], dtype=torch.long, device=ncrna_feat.device)
        logits, probs, _ = model(
            ncrna_feat,
            drug_feat,
            disease_feat,
            ncrna_ids,
            drug_ids,
            disease_ids,
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights,
        )
    pred = probs.argmax(dim=1).detach().cpu().numpy()
    return pred, probs.detach().cpu().numpy()


def positive_metrics(y_true, y_pred):
    return {
        "sample_size": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[1, 2], average="macro", zero_division=0)),
    }


def summarize_positive_subset(name, subset_df):
    if subset_df.empty:
        return None
    return {
        "subset": name,
        "sample_size": int(len(subset_df)),
        "accuracy": float(accuracy_score(subset_df["label_id"], subset_df["pred_label_id"])),
        "balanced_accuracy": float(balanced_accuracy_score(subset_df["label_id"], subset_df["pred_label_id"])),
        "macro_f1": float(f1_score(subset_df["label_id"], subset_df["pred_label_id"], labels=[1, 2], average="macro", zero_division=0)),
        "mean_confidence": float(subset_df["confidence"].mean()),
        "resistance_count": int((subset_df["label_id"] == 1).sum()),
        "sensitivity_count": int((subset_df["label_id"] == 2).sum()),
    }


def assign_confidence_strata(df):
    ordered = df.sort_values(["confidence", "row_index"], ascending=[False, True]).reset_index(drop=True).copy()
    if ordered.empty:
        ordered["confidence_stratum"] = []
        return ordered
    band_ids = np.floor(np.linspace(0, 3, len(ordered), endpoint=False)).astype(int)
    band_ids = np.clip(band_ids, 0, 2)
    band_name_map = {0: "top", 1: "middle", 2: "bottom"}
    ordered["confidence_stratum"] = [band_name_map[idx] for idx in band_ids]
    return ordered


def build_confidence_summary(df):
    ordered = assign_confidence_strata(df)
    rows = []
    for band_name in ["top", "middle", "bottom"]:
        subset = ordered[ordered["confidence_stratum"] == band_name]
        if subset.empty:
            continue
        rows.append({
            "confidence_stratum": band_name,
            "sample_size": int(len(subset)),
            "mean_confidence": float(subset["confidence"].mean()),
            "accuracy": float(subset["pred_correct"].mean()),
            "resistance_count": int((subset["label_id"] == 1).sum()),
            "sensitivity_count": int((subset["label_id"] == 2).sum()),
        })
    return ordered, pd.DataFrame(rows)


def build_topk_hits(df, ks=(10, 20, 30)):
    ordered = df.sort_values(["confidence", "row_index"], ascending=[False, True]).reset_index(drop=True)
    rows = []
    for k in ks:
        subset = ordered.head(min(k, len(ordered)))
        if subset.empty:
            continue
        rows.append({
            "top_k": int(k),
            "effective_k": int(len(subset)),
            "hit_rate": float(subset["pred_correct"].mean()),
            "mean_confidence": float(subset["confidence"].mean()),
            "resistance_count": int((subset["label_id"] == 1).sum()),
            "sensitivity_count": int((subset["label_id"] == 2).sum()),
        })
    return pd.DataFrame(rows)


def build_year_summary(df):
    rows = []
    for year in sorted(int(year) for year in df["source_year"].dropna().unique()):
        subset = df[df["source_year"] == year]
        record = summarize_positive_subset(f"year_{year}", subset)
        if record is not None:
            record["year"] = year
            rows.append(record)
    return pd.DataFrame(rows)


def build_label_year_summary(df):
    rows = []
    for year in sorted(int(year) for year in df["source_year"].dropna().unique()):
        for label_id, label_name in ((1, "resistance"), (2, "sensitivity")):
            subset = df[(df["source_year"] == year) & (df["label_id"] == label_id)]
            if subset.empty:
                continue
            rows.append({
                "year": int(year),
                "label": label_name,
                "sample_size": int(len(subset)),
                "accuracy": float(subset["pred_correct"].mean()),
                "mean_confidence": float(subset["confidence"].mean()),
            })
    return pd.DataFrame(rows)


def multiclass_auc_aupr(y_true, probs):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    aupr = average_precision_score(y_bin, probs, average="macro")
    return float(auc), float(aupr)


def sample_matched_negatives(positive_df, assoc_df, ncrna_ids, drug_ids, disease_ids, rng):
    known = set((int(r[0]), int(r[1]), int(r[2])) for r in assoc_df[:, :3])
    positives = set((int(r.ncrna_entity_id), int(r.drug_entity_id), int(r.disease_entity_id)) for r in positive_df.itertuples(index=False))
    negatives = []
    for row in positive_df.itertuples(index=False):
        disease_id = int(row.disease_entity_id)
        candidates = []
        for _ in range(2000):
            triplet = (
                int(rng.choice(ncrna_ids)),
                int(rng.choice(drug_ids)),
                disease_id,
            )
            if triplet in known or triplet in positives:
                continue
            candidates.append(triplet)
            if len(candidates) >= 10:
                break
        if not candidates:
            continue
        negatives.append(candidates[0])
    return np.asarray(negatives, dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen-checkpoint external validation on prepared external triplets.")
    parser.add_argument("--prepared-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--assoc-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset-name", default="dataset2")
    parser.add_argument("--negative-repeats", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    prepared_df = pd.read_csv(args.prepared_csv)
    eligible = prepared_df[prepared_df["eligible_for_frozen_scoring"]].copy()
    eligible = eligible[eligible["label_id"].isin([1, 2])]
    if eligible.empty:
        raise ValueError("No eligible external rows found for frozen scoring.")

    assoc = np.loadtxt(args.assoc_path, dtype=int)
    ncrna_feat, drug_feat, disease_feat = load_feature_tensors(Path(args.data_dir), device)
    model = load_model(Path(args.checkpoint), (ncrna_feat.shape[1], drug_feat.shape[1], disease_feat.shape[1]), device)
    hyperedge_index, sign_weights = build_relational_hypergraph(assoc, ncrna_feat.shape[0], drug_feat.shape[0], disease_feat.shape[0], device)

    positive_triplets = eligible[["ncrna_entity_id", "drug_entity_id", "disease_entity_id"]].astype(int).to_numpy()
    y_true = eligible["label_id"].astype(int).to_numpy()
    y_pred, probs = predict_triplets(model, ncrna_feat, drug_feat, disease_feat, positive_triplets, hyperedge_index, sign_weights)

    positive_rows = []
    for idx, row in enumerate(eligible.itertuples(index=False)):
        confidence = float(probs[idx].max())
        positive_rows.append({
            "dataset": args.dataset_name,
            "row_index": int(row.row_index),
            "ncrna_name": row.ncrna_canonical_name,
            "drug_name": row.drug_canonical_name,
            "disease_name": row.disease_canonical_name,
            "label_id": int(row.label_id),
            "label": LABEL_NAMES[int(row.label_id)],
            "pred_label_id": int(y_pred[idx]),
            "pred_label": LABEL_NAMES[int(y_pred[idx])],
            "pred_correct": bool(int(y_pred[idx]) == int(row.label_id)),
            "prob_non_association": float(probs[idx, 0]),
            "prob_resistance": float(probs[idx, 1]),
            "prob_sensitivity": float(probs[idx, 2]),
            "confidence": confidence,
            "source_year": row.source_year_int if hasattr(row, "source_year_int") else row.source_year,
            "source_id": row.source_id,
            "source_name": row.source_name,
            "time_heldout_2025plus": bool(row.time_heldout_2025plus),
            "time_heldout_2026only": bool(getattr(row, "time_heldout_2026only", False)),
            "unseen_source": bool(row.unseen_source),
            "evidence_note": row.evidence_note,
        })
    positive_df = pd.DataFrame(positive_rows)
    positive_df.to_csv(outdir / "external_positive_results.csv", index=False)

    subset_summaries = []
    for subset_name, subset_df in {
        "all_external_positive": positive_df,
        "time_heldout_2025plus": positive_df[positive_df["time_heldout_2025plus"] == True],
        "time_heldout_2026only": positive_df[positive_df["time_heldout_2026only"] == True],
        "unseen_source": positive_df[positive_df["unseen_source"] == True],
    }.items():
        summary_row = summarize_positive_subset(subset_name, subset_df)
        if summary_row is not None:
            subset_summaries.append(summary_row)
    subset_summary_df = pd.DataFrame(subset_summaries)
    subset_summary_df.to_csv(outdir / "external_positive_subset_metrics.csv", index=False)

    positive_df_ranked, confidence_summary_df = build_confidence_summary(positive_df)
    positive_df_ranked.to_csv(outdir / "external_positive_ranked.csv", index=False)
    confidence_summary_df.to_csv(outdir / "external_positive_confidence_summary.csv", index=False)
    topk_df = build_topk_hits(positive_df_ranked)
    topk_df.to_csv(outdir / "external_positive_topk_hits.csv", index=False)
    year_summary_df = build_year_summary(positive_df)
    year_summary_df.to_csv(outdir / "external_positive_year_summary.csv", index=False)
    label_year_summary_df = build_label_year_summary(positive_df)
    label_year_summary_df.to_csv(outdir / "external_positive_year_label_summary.csv", index=False)

    summary = {
        "dataset": args.dataset_name,
        "mapped_sample_size": int(len(prepared_df[prepared_df["fully_mapped"]])),
        "novel_eligible_sample_size": int(len(eligible)),
        "resistance_count": int((eligible["label_id"] == 1).sum()),
        "sensitivity_count": int((eligible["label_id"] == 2).sum()),
        "all_external_positive": positive_metrics(y_true, y_pred),
    }

    for subset_name, mask in {
        "time_heldout_2025plus": eligible["time_heldout_2025plus"].fillna(False).to_numpy(dtype=bool),
        "time_heldout_2026only": eligible["time_heldout_2026only"].fillna(False).to_numpy(dtype=bool),
        "unseen_source": eligible["unseen_source"].fillna(False).to_numpy(dtype=bool),
    }.items():
        if mask.sum() == 0:
            continue
        summary[subset_name] = positive_metrics(y_true[mask], y_pred[mask])

    summary["confidence_strata"] = confidence_summary_df.to_dict(orient="records")
    summary["topk_hit_rates"] = topk_df.to_dict(orient="records")
    summary["year_summaries"] = year_summary_df.to_dict(orient="records")
    summary["year_label_summaries"] = label_year_summary_df.to_dict(orient="records")

    rng = np.random.default_rng(42)
    ncrna_ids = np.arange(ncrna_feat.shape[0])
    drug_ids = np.arange(drug_feat.shape[0])
    disease_ids = np.arange(disease_feat.shape[0])
    neg_results = []
    for repeat in range(args.negative_repeats):
        negatives = sample_matched_negatives(eligible, assoc, ncrna_ids, drug_ids, disease_ids, rng)
        if len(negatives) == 0:
            continue
        neg_pred, neg_probs = predict_triplets(model, ncrna_feat, drug_feat, disease_feat, negatives, hyperedge_index, sign_weights)
        y_ext = np.concatenate([y_true, np.zeros(len(negatives), dtype=int)])
        probs_ext = np.vstack([probs, neg_probs])
        pred_ext = np.concatenate([y_pred, neg_pred])
        auc, aupr = multiclass_auc_aupr(y_ext, probs_ext)
        neg_results.append({
            "repeat": repeat,
            "sample_size": int(len(y_ext)),
            "macro_f1": float(f1_score(y_ext, pred_ext, labels=[0, 1, 2], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_ext, pred_ext)),
            "balanced_accuracy": float(balanced_accuracy_score(y_ext, pred_ext)),
            "auc": auc,
            "aupr": aupr,
        })
    pd.DataFrame(neg_results).to_csv(outdir / "external_negative_sampling_results.csv", index=False)

    if neg_results:
        summary["negative_sampling_repeat_count"] = len(neg_results)
        summary["negative_sampling_mean"] = {
            key: float(np.mean([row[key] for row in neg_results]))
            for key in ["macro_f1", "accuracy", "balanced_accuracy", "auc", "aupr"]
        }
        summary["negative_sampling_std"] = {
            key: float(np.std([row[key] for row in neg_results], ddof=0))
            for key in ["macro_f1", "accuracy", "balanced_accuracy", "auc", "aupr"]
        }

    (outdir / "external_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
