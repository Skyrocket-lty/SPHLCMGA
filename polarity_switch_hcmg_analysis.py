import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

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

from dataset2_model import NcrnaDrugDiseaseDualBranchModel, parameters_set


TARGET_NCRNA_NAME = "hsa-miR-21-5p"
TARGET_DRUG_NAME = "Cisplatin"
TARGET_MIXED_DISEASES = {
    "Esophageal Cancer",
    "Hepatocellular Carcinoma",
    "Ovarian Cancer",
}


def cliffs_delta(x, y):
    greater = 0
    lower = 0
    for a in x:
        greater += np.sum(a > y)
        lower += np.sum(a < y)
    return float((greater - lower) / (len(x) * len(y)))


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
    sen_edge, sen_sign = build_single(sensitive_data, 1.0)

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


def load_mapping(mapping_path):
    mapping_path = Path(mapping_path)
    if mapping_path.suffix.lower() == '.csv':
        df = pd.read_csv(mapping_path)
    else:
        df = pd.read_excel(mapping_path)
    ncrna_map = {str(name): int(entity_id) for name, entity_id in zip(df[df["type"] == "ncRNA"]["name"], df[df["type"] == "ncRNA"]["id"])}
    drug_map = {str(name): int(entity_id) for name, entity_id in zip(df[df["type"] == "drug"]["name"], df[df["type"] == "drug"]["id"])}
    disease_df = df[df["type"].isin(["cancer", "disease"])]
    disease_name = {int(entity_id): str(name) for name, entity_id in zip(disease_df["name"], disease_df["id"])}
    return ncrna_map, drug_map, disease_name


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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_row(model, ncrna_feat, drug_feat, disease_feat, hyperedge_index, sign_weights, ncrna_id, drug_id, disease_id):
    with torch.no_grad():
        logits, probs, _, explain = model(
            ncrna_feat,
            drug_feat,
            disease_feat,
            torch.tensor([ncrna_id], dtype=torch.long, device=ncrna_feat.device),
            torch.tensor([drug_id], dtype=torch.long, device=ncrna_feat.device),
            torch.tensor([disease_id], dtype=torch.long, device=ncrna_feat.device),
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights,
            return_explain=True,
        )
    cross = explain["cross_attention"]
    nc_dg = torch.linalg.norm(cross["nc_dg_gated"], dim=1).item()
    nc_dis = torch.linalg.norm(cross["nc_dis_gated"], dim=1).item()
    dg_dis = torch.linalg.norm(cross["dg_dis_gated"], dim=1).item()
    total = nc_dg + nc_dis + dg_dis + 1e-12
    return {
        "pred_label": int(probs.argmax(dim=1).item()),
        "pred_resistance": float(probs[0, 1].item()),
        "pred_sensitivity": float(probs[0, 2].item()),
        "pred_non_association": float(probs[0, 0].item()),
        "nc_dg_contribution": float(nc_dg / total),
        "nc_dis_contribution": float(nc_dis / total),
        "dg_dis_contribution": float(dg_dis / total),
        "gate_nc_dg_mean": float(cross["gate_nc_dg"].mean().item()),
        "gate_nc_dis_mean": float(cross["gate_nc_dis"].mean().item()),
        "gate_dg_dis_mean": float(cross["gate_dg_dis"].mean().item()),
        "gate_nc_dg_var": float(cross["gate_nc_dg"].var(unbiased=False).item()),
        "gate_nc_dis_var": float(cross["gate_nc_dis"].var(unbiased=False).item()),
        "gate_dg_dis_var": float(cross["gate_dg_dis"].var(unbiased=False).item()),
    }


def build_stats(df, analysis_scope):
    metrics = [
        "nc_dg_contribution",
        "nc_dis_contribution",
        "dg_dis_contribution",
        "gate_nc_dg_mean",
        "gate_nc_dis_mean",
        "gate_dg_dis_mean",
        "gate_nc_dg_var",
        "gate_nc_dis_var",
        "gate_dg_dis_var",
    ]
    rows = []
    resistance = df[df["label_name"] == "resistance"]
    sensitivity = df[df["label_name"] == "sensitivity"]
    for metric in metrics:
        x = resistance[metric].to_numpy(dtype=float)
        y = sensitivity[metric].to_numpy(dtype=float)
        stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "metric": metric,
                "resistance_n": int(len(x)),
                "sensitivity_n": int(len(y)),
                "resistance_mean": float(np.mean(x)),
                "sensitivity_mean": float(np.mean(y)),
                "mean_difference": float(np.mean(x) - np.mean(y)),
                "mannwhitney_u": float(stat),
                "p_value": float(p_value),
                "cliffs_delta": float(cliffs_delta(x, y)),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze HCMG polarity-switch distributions for hsa-miR-21-5p–Cisplatin.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--assoc-path", required=True)
    parser.add_argument("--mapping-xlsx", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    ncrna_feat = ((np.loadtxt(Path(args.data_dir) / "LLM_rna_sim.txt") + np.loadtxt(Path(args.data_dir) / "RNA.txt")) / 2.0).astype(np.float32)
    drug_feat = ((np.loadtxt(Path(args.data_dir) / "LLM_drug_sim.txt") + np.loadtxt(Path(args.data_dir) / "drug.txt")) / 2.0).astype(np.float32)
    disease_feat = ((np.loadtxt(Path(args.data_dir) / "LLM_disease_sim.txt") + np.loadtxt(Path(args.data_dir) / "disease.txt")) / 2.0).astype(np.float32)
    assoc = np.loadtxt(args.assoc_path, dtype=int)

    ncrna_map, drug_map, disease_name = load_mapping(args.mapping_xlsx)
    target_ncrna_id = ncrna_map[TARGET_NCRNA_NAME]
    target_drug_id = drug_map[TARGET_DRUG_NAME]

    hyperedge_index, sign_weights = build_relational_hypergraph(assoc, ncrna_feat.shape[0], drug_feat.shape[0], disease_feat.shape[0], device)
    model = load_model(args.checkpoint, (ncrna_feat.shape[1], drug_feat.shape[1], disease_feat.shape[1]), device)

    ncrna_feat_t = torch.from_numpy(ncrna_feat).to(device)
    drug_feat_t = torch.from_numpy(drug_feat).to(device)
    disease_feat_t = torch.from_numpy(disease_feat).to(device)

    rows = assoc[(assoc[:, 0] == target_ncrna_id) & (assoc[:, 1] == target_drug_id) & np.isin(assoc[:, 3], [1, 2])]
    rows = np.unique(rows[:, [2, 3]].astype(int), axis=0)
    records = []
    disease_counter = defaultdict(Counter)
    for row in rows:
        disease_id, label = row.astype(int)
        result = compute_row(model, ncrna_feat_t, drug_feat_t, disease_feat_t, hyperedge_index, sign_weights, target_ncrna_id, target_drug_id, disease_id)
        record = {
            "ncrna_name": TARGET_NCRNA_NAME,
            "drug_name": TARGET_DRUG_NAME,
            "disease_id": int(disease_id),
            "disease_name": disease_name[int(disease_id)],
            "label": int(label),
            "label_name": "resistance" if int(label) == 1 else "sensitivity",
            **result,
        }
        disease_counter[int(disease_id)][int(label)] += 1
        records.append(record)

    record_df = pd.DataFrame(records)
    record_df.to_csv(outdir / "miR21_cisplatin_hcmg_records.csv", index=False)

    summary_rows = []
    for disease_id, counter in disease_counter.items():
        status = "mixed" if len(counter) > 1 else ("resistance" if 1 in counter else "sensitivity")
        summary_rows.append(
            {
                "disease_id": disease_id,
                "disease_name": disease_name[disease_id],
                "resistance_count": int(counter.get(1, 0)),
                "sensitivity_count": int(counter.get(2, 0)),
                "support": int(counter.get(1, 0) + counter.get(2, 0)),
                "polarity_status": status,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["support", "disease_name"], ascending=[False, True])
    summary_df.to_csv(outdir / "miR21_cisplatin_disease_polarity_summary.csv", index=False)

    all_stats = build_stats(record_df, "all_samples")
    nonmixed_df = record_df[~record_df["disease_name"].isin(TARGET_MIXED_DISEASES)].copy()
    nonmixed_stats = build_stats(nonmixed_df, "exclude_mixed_diseases")
    stats_df = pd.concat([all_stats, nonmixed_stats], ignore_index=True)
    stats_df.to_csv(outdir / "miR21_cisplatin_hcmg_stats.csv", index=False)

    payload = {
        "target_pair": f"{TARGET_NCRNA_NAME} | {TARGET_DRUG_NAME}",
        "resistance_n": int((record_df['label_name'] == 'resistance').sum()),
        "sensitivity_n": int((record_df['label_name'] == 'sensitivity').sum()),
        "mixed_diseases": sorted(TARGET_MIXED_DISEASES),
        "selected_resistance_contexts": ["Non-Small Cell Lung Cancer", "Lung Adenocarcinoma"],
        "selected_sensitivity_contexts": ["Pancreatic Cancer", "Small Cell Lung Cancer"],
    }
    (outdir / "miR21_cisplatin_hcmg_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
