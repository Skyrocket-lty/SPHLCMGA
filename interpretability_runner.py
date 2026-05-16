import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from maintest import build_relational_hypergraph, get_train_val_data, random_seed
from model import NcrnaDrugDiseaseDualBranchModel, parameters_set


CLASS_NAME = {
    0: "non-association",
    1: "resistance",
    2: "sensitivity",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Interpretability analysis runner for SPHLCMGA")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="interpretability_results")
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def to_numpy_tensor(array, device):
    return torch.from_numpy(array).float().to(device)


def load_dataset(project_root, dataset):
    root = Path(project_root)
    if dataset == "dataset1":
        data_dir = root / "Data"
        ncrna_base = np.loadtxt(data_dir / "ncRNA.txt")
        drug_base = np.loadtxt(data_dir / "drug.txt")
        disease_base = np.loadtxt(data_dir / "disease.txt")
        ncrna_llm = np.loadtxt(data_dir / "LLM_rna_sim.txt")
        drug_llm = np.loadtxt(data_dir / "LLM_drug_sim.txt")
        disease_llm = np.loadtxt(data_dir / "LLM_disease_sim.txt")
        adj_data = np.loadtxt(data_dir / "association3.txt")
    else:
        data_dir = root / "Data2"
        ncrna_base = np.loadtxt(data_dir / "RNA.txt")
        drug_base = np.loadtxt(data_dir / "drug.txt")
        disease_base = np.loadtxt(data_dir / "disease.txt")
        ncrna_llm = np.loadtxt(data_dir / "LLM_rna_sim.txt")
        drug_llm = np.loadtxt(data_dir / "LLM_drug_sim.txt")
        disease_llm = np.loadtxt(data_dir / "LLM_disease_sim.txt")
        adj_data = np.loadtxt(data_dir / "association.txt")

    ncrna_sim = ((ncrna_base + ncrna_llm) / 2.0).astype(np.float32)
    drug_sim = ((drug_base + drug_llm) / 2.0).astype(np.float32)
    disease_sim = ((disease_base + disease_llm) / 2.0).astype(np.float32)
    adj_data = np.atleast_2d(adj_data).astype(int)
    return ncrna_sim, drug_sim, disease_sim, adj_data


def build_model(ncrna_feat, drug_feat, disease_feat, device):
    args = parameters_set()
    model = NcrnaDrugDiseaseDualBranchModel(
        ncrna_dim=ncrna_feat.shape[1],
        drug_dim=drug_feat.shape[1],
        disease_dim=disease_feat.shape[1],
        modal_hidden_dim=args.modal_hidden_dim,
        modal_out_dim=args.modal_out_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        hgnn_dim_1=args.hgnn_dim_1,
        dropout=args.dropout,
    ).to(device)
    return model, args


def predict_dataset(model, ncrna_feat, drug_feat, disease_feat, data, hyperedge_index, sign_weights):
    model.eval()
    with torch.no_grad():
        ncrna_ids = torch.tensor(data[:, 0], dtype=torch.long, device=ncrna_feat.device)
        drug_ids = torch.tensor(data[:, 1], dtype=torch.long, device=ncrna_feat.device)
        disease_ids = torch.tensor(data[:, 2], dtype=torch.long, device=ncrna_feat.device)
        logits, pred_probs, _, explain_outputs = model(
            ncrna_feat,
            drug_feat,
            disease_feat,
            ncrna_ids,
            drug_ids,
            disease_ids,
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights,
            return_explain=True,
        )
    preds = pred_probs.argmax(dim=1).detach().cpu().numpy()
    probs = pred_probs.detach().cpu().numpy()
    branch1_norm = torch.linalg.norm(explain_outputs["branch1_feat"], dim=1).detach().cpu().numpy()
    branch2_norm = torch.linalg.norm(explain_outputs["branch2_feat"], dim=1).detach().cpu().numpy()

    cross_details = explain_outputs["cross_attention"]
    nc_dg = torch.linalg.norm(cross_details["nc_dg_gated"], dim=1).detach().cpu().numpy()
    nc_dis = torch.linalg.norm(cross_details["nc_dis_gated"], dim=1).detach().cpu().numpy()
    dg_dis = torch.linalg.norm(cross_details["dg_dis_gated"], dim=1).detach().cpu().numpy()
    contrib = np.stack([nc_dg, nc_dis, dg_dis], axis=1)
    contrib_sum = contrib.sum(axis=1, keepdims=True) + 1e-12
    contrib = contrib / contrib_sum
    return preds, probs, branch1_norm, branch2_norm, contrib, explain_outputs


def build_records(dataset, fold_num, data, preds, probs, branch1_norm, branch2_norm, contrib):
    records = []
    labels = data[:, 3].astype(int)
    for index, row in enumerate(data):
        prob = probs[index]
        label = int(labels[index])
        pred = int(preds[index])
        sorted_idx = np.argsort(prob)
        second_idx = int(sorted_idx[-2])
        record = {
            "dataset": dataset,
            "fold": fold_num,
            "sample_index": int(index),
            "ncrna_id": int(row[0]),
            "drug_id": int(row[1]),
            "disease_id": int(row[2]),
            "label": label,
            "label_name": CLASS_NAME[label],
            "pred": pred,
            "pred_name": CLASS_NAME[pred],
            "correct": bool(label == pred),
            "true_prob": float(prob[label]),
            "margin": float(prob[label] - prob[second_idx]),
            "probs": [float(x) for x in prob.tolist()],
            "branch1_norm": float(branch1_norm[index]),
            "branch2_norm": float(branch2_norm[index]),
            "nc_dg_contrib": float(contrib[index, 0]),
            "nc_dis_contrib": float(contrib[index, 1]),
            "dg_dis_contrib": float(contrib[index, 2]),
        }
        records.append(record)
    return records


def choose_representatives(records):
    grouped = {label: [] for label in CLASS_NAME}
    for record in records:
        if record["correct"]:
            grouped[record["label"]].append(record)

    selected = {}
    for label in [2, 1]:
        candidates = sorted(
            grouped[label],
            key=lambda item: (item["true_prob"], item["margin"]),
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"No correct representative sample found for class {CLASS_NAME[label]}")
        selected[label] = candidates[0]

    preferred_diseases = {selected[2]["disease_id"], selected[1]["disease_id"]}
    non_assoc_candidates = sorted(
        grouped[0],
        key=lambda item: (item["disease_id"] in preferred_diseases, item["true_prob"], item["margin"]),
        reverse=True,
    )
    if not non_assoc_candidates:
        raise RuntimeError("No correct representative sample found for class non-association")
    selected[0] = non_assoc_candidates[0]
    return {CLASS_NAME[key]: value for key, value in selected.items()}


def top_candidates(records, top_k=5):
    topk = {}
    for label in [2, 1, 0]:
        candidates = [item for item in records if item["label"] == label and item["correct"]]
        candidates = sorted(candidates, key=lambda item: (item["true_prob"], item["margin"]), reverse=True)
        if len(candidates) < top_k:
            raise RuntimeError(f"Class {CLASS_NAME[label]} has fewer than {top_k} correct candidates")
        topk[CLASS_NAME[label]] = candidates[:top_k]
    return topk


def compute_fold_f1(records):
    labels = [item["label"] for item in records]
    preds = [item["pred"] for item in records]
    return float(f1_score(labels, preds, labels=[0, 1, 2], average="macro", zero_division=0))


def compute_entity_bias(train_pos, ncrna_num, drug_num, disease_num):
    ncrna_degree = np.zeros(ncrna_num, dtype=int)
    drug_degree = np.zeros(drug_num, dtype=int)
    disease_degree = np.zeros(disease_num, dtype=int)
    ncrna_seen = np.zeros(ncrna_num, dtype=int)
    drug_seen = np.zeros(drug_num, dtype=int)
    disease_seen = np.zeros(disease_num, dtype=int)

    for row in train_pos:
        sign = 1 if int(row[3]) == 2 else -1
        n_id = int(row[0])
        d_id = int(row[1])
        dis_id = int(row[2])
        ncrna_degree[n_id] += sign
        drug_degree[d_id] += sign
        disease_degree[dis_id] += sign
        ncrna_seen[n_id] += 1
        drug_seen[d_id] += 1
        disease_seen[dis_id] += 1

    def label_array(degree_array, seen_array):
        labels = []
        for degree, seen in zip(degree_array, seen_array):
            if seen == 0:
                labels.append("isolated")
            elif degree > 0:
                labels.append("sensitivity-biased")
            elif degree < 0:
                labels.append("resistance-biased")
            else:
                labels.append("balanced")
        return np.asarray(labels)

    return {
        "ncrna": label_array(ncrna_degree, ncrna_seen),
        "drug": label_array(drug_degree, drug_seen),
        "disease": label_array(disease_degree, disease_seen),
    }


def export_top_candidates(output_dir, dataset, candidates):
    path = output_dir / f"{dataset}_top_candidates.csv"
    fieldnames = [
        "dataset", "class", "rank", "fold", "ncrna_id", "drug_id", "disease_id",
        "true_prob", "margin", "pred_probs", "branch1_norm", "branch2_norm",
        "nc_dg_contrib", "nc_dis_contrib", "dg_dis_contrib"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, rows in candidates.items():
            for rank, row in enumerate(rows, start=1):
                writer.writerow({
                    "dataset": dataset,
                    "class": class_name,
                    "rank": rank,
                    "fold": row["fold"],
                    "ncrna_id": row["ncrna_id"],
                    "drug_id": row["drug_id"],
                    "disease_id": row["disease_id"],
                    "true_prob": f"{row['true_prob']:.6f}",
                    "margin": f"{row['margin']:.6f}",
                    "pred_probs": json.dumps(row["probs"]),
                    "branch1_norm": f"{row['branch1_norm']:.6f}",
                    "branch2_norm": f"{row['branch2_norm']:.6f}",
                    "nc_dg_contrib": f"{row['nc_dg_contrib']:.6f}",
                    "nc_dis_contrib": f"{row['nc_dis_contrib']:.6f}",
                    "dg_dis_contrib": f"{row['dg_dis_contrib']:.6f}",
                })


def export_branch_norms(output_dir, dataset, records):
    path = output_dir / f"{dataset}_branch_norms.csv"
    fieldnames = ["dataset", "fold", "label", "correct", "branch1_norm", "branch2_norm"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "dataset": dataset,
                "fold": row["fold"],
                "label": row["label_name"],
                "correct": int(row["correct"]),
                "branch1_norm": f"{row['branch1_norm']:.6f}",
                "branch2_norm": f"{row['branch2_norm']:.6f}",
            })


def export_embedding_payload(output_dir, dataset, fold_num, pre_embedding, signed_embedding, unsigned_embedding,
                            labels_by_type):
    entity_type = []
    bias_label = []
    for key in ["ncrna", "drug", "disease"]:
        entity_type.extend([key] * len(labels_by_type[key]))
        bias_label.extend(labels_by_type[key].tolist())

    np.savez_compressed(
        output_dir / f"{dataset}_embedding_payload.npz",
        best_fold=np.asarray([fold_num], dtype=np.int64),
        pre_embedding=pre_embedding,
        signed_embedding=signed_embedding,
        unsigned_embedding=unsigned_embedding,
        entity_type=np.asarray(entity_type),
        bias_label=np.asarray(bias_label),
    )


def build_triplet_label(record):
    return f"RNA {record['ncrna_id']} | Drug {record['drug_id']} | Disease {record['disease_id']}"


def main():
    cli_args = parse_args()
    project_root = Path(cli_args.project_root).resolve()
    output_dir = (project_root / cli_args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cli_args.device if torch.cuda.is_available() else "cpu")
    random_seed(cli_args.seed)

    ncrna_sim, drug_sim, disease_sim, adj_data = load_dataset(project_root, cli_args.dataset)
    ncrna_feat = to_numpy_tensor(ncrna_sim, device)
    drug_feat = to_numpy_tensor(drug_sim, device)
    disease_feat = to_numpy_tensor(disease_sim, device)

    model, args = build_model(ncrna_feat, drug_feat, disease_feat, device)
    kf = __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(
        n_splits=args.k_fold, shuffle=True, random_state=args.seed
    )

    all_records = []
    fold_cache = {}
    fold_scores = []
    for fold_num, (train_index, val_index) in enumerate(kf.split(adj_data), start=1):
        train_pos, train_neg, val_pos, val_neg = get_train_val_data(adj_data, train_index, val_index, adj_data, args.seed)
        train_data = np.vstack([train_pos, train_neg]) if len(train_neg) > 0 else train_pos
        val_data = np.vstack([val_pos, val_neg]) if len(val_neg) > 0 else val_pos

        hyperedge_index, sign_weights = build_relational_hypergraph(
            train_data, ncrna_feat.shape[0], drug_feat.shape[0], disease_feat.shape[0], device
        )

        checkpoint_path = project_root / "Data" / "best_models" / f"best_model_fold_{fold_num}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        preds, probs, branch1_norm, branch2_norm, contrib, _ = predict_dataset(
            model, ncrna_feat, drug_feat, disease_feat, val_data, hyperedge_index, sign_weights
        )
        fold_records = build_records(
            cli_args.dataset, fold_num, val_data, preds, probs, branch1_norm, branch2_norm, contrib
        )
        all_records.extend(fold_records)
        fold_f1 = compute_fold_f1(fold_records)
        fold_scores.append({"fold": fold_num, "f1_macro": fold_f1})
        fold_cache[fold_num] = {
            "train_pos": train_pos,
            "train_data": train_data,
            "val_data": val_data,
            "hyperedge_index": hyperedge_index,
            "sign_weights": sign_weights,
        }

    representatives = choose_representatives(all_records)
    candidate_table = top_candidates(all_records, top_k=5)
    export_top_candidates(output_dir, cli_args.dataset, candidate_table)
    export_branch_norms(output_dir, cli_args.dataset, all_records)

    representative_payload = {}
    for class_name, record in representatives.items():
        fold_num = record["fold"]
        fold_state = fold_cache[fold_num]
        checkpoint_path = project_root / "Data" / "best_models" / f"best_model_fold_{fold_num}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        sample = np.array([[record["ncrna_id"], record["drug_id"], record["disease_id"], record["label"]]], dtype=int)
        _, _, _, explain_outputs = model(
            ncrna_feat,
            drug_feat,
            disease_feat,
            torch.tensor(sample[:, 0], dtype=torch.long, device=device),
            torch.tensor(sample[:, 1], dtype=torch.long, device=device),
            torch.tensor(sample[:, 2], dtype=torch.long, device=device),
            hyperedge_index=fold_state["hyperedge_index"],
            sign_weights=fold_state["sign_weights"],
            return_explain=True,
        )
        cross_details = explain_outputs["cross_attention"]
        contributions = np.array([
            torch.linalg.norm(cross_details["nc_dg_gated"], dim=1).item(),
            torch.linalg.norm(cross_details["nc_dis_gated"], dim=1).item(),
            torch.linalg.norm(cross_details["dg_dis_gated"], dim=1).item(),
        ], dtype=np.float64)
        contributions = (contributions / (contributions.sum() + 1e-12)).tolist()
        representative_payload[class_name] = {
            "fold": fold_num,
            "triplet_label": build_triplet_label(record),
            "ncrna_id": record["ncrna_id"],
            "drug_id": record["drug_id"],
            "disease_id": record["disease_id"],
            "label": record["label_name"],
            "pred_probs": record["probs"],
            "true_prob": record["true_prob"],
            "margin": record["margin"],
            "interaction_contributions": {
                "ncRNA-drug": contributions[0],
                "ncRNA-disease": contributions[1],
                "drug-disease": contributions[2],
            },
        }

    best_fold = max(fold_scores, key=lambda item: item["f1_macro"])["fold"]
    fold_state = fold_cache[best_fold]
    checkpoint_path = project_root / "Data" / "best_models" / f"best_model_fold_{best_fold}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    probe_sample = fold_state["val_data"][:1]
    _, _, _, explain_outputs = model(
        ncrna_feat,
        drug_feat,
        disease_feat,
        torch.tensor(probe_sample[:, 0], dtype=torch.long, device=device),
        torch.tensor(probe_sample[:, 1], dtype=torch.long, device=device),
        torch.tensor(probe_sample[:, 2], dtype=torch.long, device=device),
        hyperedge_index=fold_state["hyperedge_index"],
        sign_weights=fold_state["sign_weights"],
        return_explain=True,
    )
    aligned = explain_outputs["aligned_embeddings"]
    pre_embedding = torch.cat([aligned["ncrna"], aligned["drug"], aligned["disease"]], dim=0).detach().cpu().numpy()
    signed_embedding = explain_outputs["all_entity_hgnn"].detach().cpu().numpy()
    unsigned_embedding = model.hgnn_encoder(
        x=torch.from_numpy(pre_embedding).float().to(device),
        hyperedge_index=fold_state["hyperedge_index"],
        sign_weights=fold_state["sign_weights"].abs(),
    ).detach().cpu().numpy()
    labels_by_type = compute_entity_bias(
        fold_state["train_pos"],
        ncrna_feat.shape[0],
        drug_feat.shape[0],
        disease_feat.shape[0],
    )
    export_embedding_payload(output_dir, cli_args.dataset, best_fold, pre_embedding, signed_embedding,
                            unsigned_embedding, labels_by_type)

    summary_path = output_dir / f"{cli_args.dataset}_interpretability_summary.json"
    summary_payload = {
        "dataset": cli_args.dataset,
        "best_fold": best_fold,
        "fold_scores": fold_scores,
        "representatives": representative_payload,
        "selection_policy": {
            "positive_classes": ["sensitivity", "resistance"],
            "negative_class": "non-association",
            "tie_break": "margin",
        },
        "notes": [
            "The current model fuses low-level and LLM-derived similarities before interaction modeling.",
            "Accordingly, the visualization reflects cross-entity interaction importance rather than raw modality-level attention weights.",
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
