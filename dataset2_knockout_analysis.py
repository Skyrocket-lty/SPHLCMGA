import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from dataset2_model import NcrnaDrugDiseaseDualBranchModel, parameters_set


LABEL_NAMES = {
    0: "non-association",
    1: "resistance",
    2: "sensitivity",
}


CASES = [
    {
        "case_id": "uca1_cisplatin_lung_cancer",
        "role": "headline",
        "source": "Table 3 predicted triplet",
        "ncrna_id": 831,
        "ncrna_name": "UCA1",
        "drug_id": 7,
        "drug_name": "Cisplatin",
        "disease_id": 58,
        "disease_name": "Lung Cancer",
        "target_ncrna_id": 831,
    },
    {
        "case_id": "malat1_cisplatin_lung_cancer",
        "role": "headline",
        "source": "Table 3 predicted triplet",
        "ncrna_id": 732,
        "ncrna_name": "MALAT1",
        "drug_id": 7,
        "drug_name": "Cisplatin",
        "disease_id": 58,
        "disease_name": "Lung Cancer",
        "target_ncrna_id": 732,
    },
    {
        "case_id": "uca1_cisplatin_nsclc",
        "role": "control",
        "source": "Known resistance control",
        "ncrna_id": 831,
        "ncrna_name": "UCA1",
        "drug_id": 7,
        "drug_name": "Cisplatin",
        "disease_id": 0,
        "disease_name": "Non-Small Cell Lung Cancer",
        "target_ncrna_id": 831,
    },
    {
        "case_id": "uca1_cisplatin_luad",
        "role": "control",
        "source": "Known resistance control",
        "ncrna_id": 831,
        "ncrna_name": "UCA1",
        "drug_id": 7,
        "drug_name": "Cisplatin",
        "disease_id": 20,
        "disease_name": "Lung Adenocarcinoma",
        "target_ncrna_id": 831,
    },
]


EXCLUDED_CASES = [
    {
        "case_id": "malat1_cisplatin_nsclc",
        "reason": "Conflicting labels in dataset2 (both resistance and sensitivity).",
        "ncrna_id": 732,
        "drug_id": 7,
        "disease_id": 0,
    }
]


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_relational_hypergraph(train_data, ncrna_num, drug_num, disease_num, device):
    resistant_data = train_data[train_data[:, 3] == 1]
    sensitive_data = train_data[train_data[:, 3] == 2]

    def build_single_hyperedge(data, sign):
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
        hyperedge_index = np.array(hyperedge_index).T
        sign_weights = np.full(len(hyperedge_nodes), sign, dtype=np.float32)
        return hyperedge_index, sign_weights

    res_edge, res_sign = build_single_hyperedge(resistant_data, sign=-1.0)
    sen_edge, sen_sign = build_single_hyperedge(sensitive_data, sign=1.0)

    if res_edge.size == 0 and sen_edge.size == 0:
        hyperedge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        sign_weights = torch.empty((0,), dtype=torch.float32, device=device)
    elif res_edge.size == 0:
        hyperedge_index = torch.tensor(sen_edge, dtype=torch.long, device=device)
        sign_weights = torch.tensor(sen_sign, dtype=torch.float32, device=device)
    elif sen_edge.size == 0:
        hyperedge_index = torch.tensor(res_edge, dtype=torch.long, device=device)
        sign_weights = torch.tensor(res_sign, dtype=torch.float32, device=device)
    else:
        res_edge[1, :] += len(sen_sign)
        hyperedge_index = np.hstack([sen_edge, res_edge])
        sign_weights = np.hstack([sen_sign, res_sign])
        hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.long, device=device)
        sign_weights = torch.tensor(sign_weights, dtype=torch.float32, device=device)
    return hyperedge_index, sign_weights


def load_feature_tensors(data_dir, device):
    ncrna_feat = (
        np.loadtxt(data_dir / "LLM_rna_sim.txt") + np.loadtxt(data_dir / "RNA.txt")
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


def load_model(checkpoint_path, ncrna_dim, drug_dim, disease_dim, device):
    args = parameters_set()
    model = NcrnaDrugDiseaseDualBranchModel(
        ncrna_dim=ncrna_dim,
        drug_dim=drug_dim,
        disease_dim=disease_dim,
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
    return model, checkpoint


def apply_feature_knockout(ncrna_feat, target_ncrna_id):
    knocked = ncrna_feat.clone()
    knocked[target_ncrna_id] = 0.0
    return knocked


def apply_graph_knockout(all_assoc, target_ncrna_id):
    return all_assoc[all_assoc[:, 0].astype(int) != int(target_ncrna_id)]


def classify_triplet_status(all_assoc, case):
    mask = (
        (all_assoc[:, 0].astype(int) == int(case["ncrna_id"]))
        & (all_assoc[:, 1].astype(int) == int(case["drug_id"]))
        & (all_assoc[:, 2].astype(int) == int(case["disease_id"]))
    )
    matched = all_assoc[mask]
    if len(matched) == 0:
        return "unknown", []
    labels = sorted(set(matched[:, 3].astype(int).tolist()))
    if len(labels) == 1:
        return f"known_{LABEL_NAMES[labels[0]]}", labels
    return "conflicted", labels


def predict_single_case(model, ncrna_feat, drug_feat, disease_feat, case, hyperedge_index, sign_weights):
    with torch.no_grad():
        torch.use_deterministic_algorithms(False)
        ncrna_ids = torch.tensor([case["ncrna_id"]], dtype=torch.long, device=ncrna_feat.device)
        drug_ids = torch.tensor([case["drug_id"]], dtype=torch.long, device=ncrna_feat.device)
        disease_ids = torch.tensor([case["disease_id"]], dtype=torch.long, device=ncrna_feat.device)
        logits, pred_probs, _, explain = model(
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

    probs = pred_probs[0].detach().cpu().numpy()
    pred_label = int(np.argmax(probs))
    cross_details = explain["cross_attention"]
    nc_dg = torch.linalg.norm(cross_details["nc_dg_gated"], dim=1).item()
    nc_dis = torch.linalg.norm(cross_details["nc_dis_gated"], dim=1).item()
    dg_dis = torch.linalg.norm(cross_details["dg_dis_gated"], dim=1).item()
    total = nc_dg + nc_dis + dg_dis + 1e-12

    return {
        "pred_label": pred_label,
        "pred_name": LABEL_NAMES[pred_label],
        "pred_confidence": float(probs[pred_label]),
        "prob_non_association": float(probs[0]),
        "prob_resistance": float(probs[1]),
        "prob_sensitivity": float(probs[2]),
        "branch1_norm": float(torch.linalg.norm(explain["branch1_feat"], dim=1).item()),
        "branch2_norm": float(torch.linalg.norm(explain["branch2_feat"], dim=1).item()),
        "interaction_contribution": {
            "ncRNA-drug": float(nc_dg / total),
            "ncRNA-disease": float(nc_dis / total),
            "drug-disease": float(dg_dis / total),
        },
    }


def evaluate_modes(model, base_ncrna_feat, drug_feat, disease_feat, all_assoc, case, device):
    ncrna_num = base_ncrna_feat.shape[0]
    drug_num = drug_feat.shape[0]
    disease_num = disease_feat.shape[0]
    modes = {
        "baseline": (False, False),
        "feature_knockout": (True, False),
        "graph_knockout": (False, True),
        "dual_knockout": (True, True),
    }

    results = {}
    for mode_name, (feature_ko, graph_ko) in modes.items():
        ncrna_feat = apply_feature_knockout(base_ncrna_feat, case["target_ncrna_id"]) if feature_ko else base_ncrna_feat
        assoc = apply_graph_knockout(all_assoc, case["target_ncrna_id"]) if graph_ko else all_assoc
        hyperedge_index, sign_weights = build_relational_hypergraph(
            assoc,
            ncrna_num,
            drug_num,
            disease_num,
            device,
        )
        mode_result = predict_single_case(
            model,
            ncrna_feat,
            drug_feat,
            disease_feat,
            case,
            hyperedge_index,
            sign_weights,
        )
        mode_result["hyperedge_count"] = int(sign_weights.numel())
        results[mode_name] = mode_result
    return results


def make_csv_rows(case, triplet_status, matched_labels, mode_results):
    rows = []
    baseline = mode_results["baseline"]
    for mode_name, result in mode_results.items():
        rows.append({
            "case_id": case["case_id"],
            "role": case["role"],
            "source": case["source"],
            "triplet": f"{case['ncrna_name']} | {case['drug_name']} | {case['disease_name']}",
            "ncrna_name": case["ncrna_name"],
            "drug_name": case["drug_name"],
            "disease_name": case["disease_name"],
            "triplet_status": triplet_status,
            "matched_labels": ";".join(LABEL_NAMES[label] for label in matched_labels) if matched_labels else "",
            "mode": mode_name,
            "pred_label": result["pred_name"],
            "pred_confidence": f"{result['pred_confidence']:.6f}",
            "prob_non_association": f"{result['prob_non_association']:.6f}",
            "prob_resistance": f"{result['prob_resistance']:.6f}",
            "prob_sensitivity": f"{result['prob_sensitivity']:.6f}",
            "delta_resistance_vs_baseline": f"{result['prob_resistance'] - baseline['prob_resistance']:.6f}",
            "delta_sensitivity_vs_baseline": f"{result['prob_sensitivity'] - baseline['prob_sensitivity']:.6f}",
            "branch1_norm": f"{result['branch1_norm']:.6f}",
            "branch2_norm": f"{result['branch2_norm']:.6f}",
            "ncRNA_drug_contribution": f"{result['interaction_contribution']['ncRNA-drug']:.6f}",
            "ncRNA_disease_contribution": f"{result['interaction_contribution']['ncRNA-disease']:.6f}",
            "drug_disease_contribution": f"{result['interaction_contribution']['drug-disease']:.6f}",
            "hyperedge_count": result["hyperedge_count"],
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run dataset2 case-specific in silico knockout analysis.")
    parser.add_argument("--work-dir", default=".", help="Dataset2 project root containing Data and Data2.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model_fold_2.pth")
    parser.add_argument("--output-dir", default="knockout_outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    work_dir = Path(args.work_dir).resolve()
    data_dir = work_dir / "Data2"
    output_dir = Path(args.output_dir).resolve() if Path(args.output_dir).is_absolute() else (work_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    ncrna_feat, drug_feat, disease_feat = load_feature_tensors(data_dir, device)
    all_assoc = np.loadtxt(data_dir / "association.txt", dtype=int)
    checkpoint_path = work_dir / args.checkpoint
    model, checkpoint = load_model(
        checkpoint_path,
        ncrna_feat.shape[1],
        drug_feat.shape[1],
        disease_feat.shape[1],
        device,
    )

    rows = []
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "checkpoint_note": "Using the same best_model_fold_2 checkpoint referenced by dataset2 case-study inference.",
        "excluded_cases": EXCLUDED_CASES,
        "cases": {},
    }

    for case in CASES:
        triplet_status, matched_labels = classify_triplet_status(all_assoc, case)
        mode_results = evaluate_modes(model, ncrna_feat, drug_feat, disease_feat, all_assoc, case, device)
        rows.extend(make_csv_rows(case, triplet_status, matched_labels, mode_results))

        baseline = mode_results["baseline"]
        dual = mode_results["dual_knockout"]
        summary["cases"][case["case_id"]] = {
            "meta": {
                "role": case["role"],
                "source": case["source"],
                "triplet": f"{case['ncrna_name']} | {case['drug_name']} | {case['disease_name']}",
                "triplet_status": triplet_status,
                "matched_labels": [LABEL_NAMES[label] for label in matched_labels],
            },
            "baseline": baseline,
            "feature_knockout": mode_results["feature_knockout"],
            "graph_knockout": mode_results["graph_knockout"],
            "dual_knockout": dual,
            "dual_delta": {
                "resistance": float(dual["prob_resistance"] - baseline["prob_resistance"]),
                "sensitivity": float(dual["prob_sensitivity"] - baseline["prob_sensitivity"]),
                "hard_switch_to_sensitivity": bool(
                    baseline["pred_name"] == "resistance" and dual["pred_name"] == "sensitivity"
                ),
            },
        }

    write_csv(output_dir / "dataset2_knockout_results.csv", rows)
    (output_dir / "dataset2_knockout_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
