import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from maintest import FocalLoss, build_relational_hypergraph, random_seed, test, train
from model import NcrnaDrugDiseaseDualBranchModel, parameters_set


DATASET_CONFIGS = {
    "dataset1": {
        "data_dir": "Data",
        "assoc_file": "association3.txt",
        "ncrna_file": "ncRNA.txt",
        "drug_file": "drug.txt",
        "disease_file": "disease.txt",
        "ncrna_llm_file": "LLM_rna_sim.txt",
        "drug_llm_file": "LLM_drug_sim.txt",
        "disease_llm_file": "LLM_disease_sim.txt",
    },
    "dataset2": {
        "data_dir": "Data2",
        "assoc_file": "association.txt",
        "ncrna_file": "RNA.txt",
        "drug_file": "drug.txt",
        "disease_file": "disease.txt",
        "ncrna_llm_file": "LLM_rna_sim.txt",
        "drug_llm_file": "LLM_drug_sim.txt",
        "disease_llm_file": "LLM_disease_sim.txt",
    },
}

AXIS_TO_COL = {"ncrna": 0, "drug": 1, "disease": 2}
DEFAULT_MONITOR_METRIC = "f1_macro"
DEFAULT_PATIENCE = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Cold-start evaluation for SPHLCMGA")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS.keys()), required=True)
    parser.add_argument("--split-axis", choices=sorted(AXIS_TO_COL.keys()), required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="cold_start_results")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--monitor-metric", default=DEFAULT_MONITOR_METRIC)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def load_inputs(project_root: Path, dataset_key: str):
    cfg = DATASET_CONFIGS[dataset_key]
    data_dir = project_root / cfg["data_dir"]

    disease_llm = np.loadtxt(data_dir / cfg["disease_llm_file"])
    drug_llm = np.loadtxt(data_dir / cfg["drug_llm_file"])
    ncrna_llm = np.loadtxt(data_dir / cfg["ncrna_llm_file"])

    disease_base = np.loadtxt(data_dir / cfg["disease_file"])
    drug_base = np.loadtxt(data_dir / cfg["drug_file"])
    ncrna_base = np.loadtxt(data_dir / cfg["ncrna_file"])

    assoc = ensure_2d(np.loadtxt(data_dir / cfg["assoc_file"]))
    assoc = assoc.astype(int)

    disease_sim = (disease_llm + disease_base) / 2.0
    drug_sim = (drug_llm + drug_base) / 2.0
    ncrna_sim = (ncrna_llm + ncrna_base) / 2.0

    return {
        "assoc": assoc,
        "ncrna_sim": ncrna_sim,
        "drug_sim": drug_sim,
        "disease_sim": disease_sim,
    }


def sample_neutral_triplets(num_samples, allowed_axis_values, split_axis, dims, positive_set, seed):
    if num_samples == 0:
        return np.empty((0, 4), dtype=int)

    rng = np.random.default_rng(seed)
    axis_col = AXIS_TO_COL[split_axis]
    allowed_axis_values = np.asarray(sorted(set(int(v) for v in allowed_axis_values)), dtype=int)
    if allowed_axis_values.size == 0:
        raise ValueError(f"No allowed {split_axis} ids are available for negative sampling.")

    negatives = []
    used = set()
    max_attempts = max(num_samples * 50, 10000)
    attempts = 0

    while len(negatives) < num_samples and attempts < max_attempts:
        attempts += 1
        triple = [0, 0, 0]
        triple[axis_col] = int(rng.choice(allowed_axis_values))
        for col, dim in enumerate(dims):
            if col != axis_col:
                triple[col] = int(rng.integers(0, dim))
        triple_key = tuple(triple)
        if triple_key in positive_set or triple_key in used:
            continue
        used.add(triple_key)
        negatives.append((*triple, 0))

    if len(negatives) < num_samples:
        raise RuntimeError(
            f"Unable to sample enough neutral triplets for {split_axis}-cold split: "
            f"requested={num_samples}, obtained={len(negatives)}"
        )

    return np.asarray(negatives, dtype=int)


def split_train_val(train_pos, split_axis, seed, val_ratio):
    groups = train_pos[:, AXIS_TO_COL[split_axis]]
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError(f"Not enough unique {split_axis} entities for train/val split.")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    inner_train_idx, val_idx = next(gss.split(train_pos, groups=groups))
    return train_pos[inner_train_idx], train_pos[val_idx]


def create_model(args_cfg, ncrna_feat, drug_feat, disease_feat, device):
    return NcrnaDrugDiseaseDualBranchModel(
        ncrna_dim=ncrna_feat.shape[1],
        drug_dim=drug_feat.shape[1],
        disease_dim=disease_feat.shape[1],
        modal_hidden_dim=args_cfg.modal_hidden_dim,
        modal_out_dim=args_cfg.modal_out_dim,
        classifier_hidden_dim=args_cfg.classifier_hidden_dim,
        hgnn_dim_1=args_cfg.hgnn_dim_1,
        dropout=args_cfg.dropout,
    ).to(device)


def save_checkpoint(path, fold_num, epoch, model, optimizer, best_metric, metric_name):
    torch.save(
        {
            "fold_num": fold_num,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "metric_name": metric_name,
        },
        path,
    )


def summarize_metrics(metric_list):
    summary = {}
    for key in metric_list[0].keys():
        if key in {"preds", "labels", "class_precision", "class_recall", "class_f1", "test_time", "memory"}:
            continue
        values = np.asarray([m[key] for m in metric_list], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def write_summary(output_dir: Path, fold_results, summary, split_axis):
    report_path = output_dir / "cold_start_results.txt"
    json_path = output_dir / "cold_start_results.json"

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"SPHLCMGA {split_axis}-cold results\n\n")
        for fold in fold_results:
            metrics = fold["metrics"]
            f.write(
                f"Fold {fold['fold_num']}: Accuracy={metrics['accuracy'] * 100:.2f}%, "
                f"Precision={metrics['precision_macro'] * 100:.2f}%, "
                f"Recall={metrics['recall_macro'] * 100:.2f}%, "
                f"F1={metrics['f1_macro'] * 100:.2f}%, "
                f"MCC={metrics['mcc'] * 100:.2f}%, "
                f"ROC_AUC={metrics['auc_ovr'] * 100:.2f}%, "
                f"AUPR={metrics['aupr_ovr'] * 100:.2f}%\n"
            )
            f.write(f"  best_epoch={fold['best_epoch']}\n")
            f.write(f"  classification_report:\n{fold['report']}\n\n")

        f.write("===== Summary (mean +- std, %) =====\n")
        ordered_keys = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "mcc",
            "auc_ovr",
            "aupr_ovr",
            "0_recall",
            "1_recall",
            "2_recall",
        ]
        for key in ordered_keys:
            stats = summary[key]
            f.write(f"{key}: {stats['mean'] * 100:.2f} +- {stats['std'] * 100:.2f}\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"fold_results": fold_results, "summary": summary}, f, indent=2)


def main():
    cli_args = parse_args()
    project_root = Path(cli_args.project_root).resolve()
    output_dir = project_root / cli_args.output_root / f"{cli_args.dataset}_{cli_args.split_axis}_cold"
    output_dir.mkdir(parents=True, exist_ok=True)

    args_cfg = parameters_set()
    if cli_args.seed is not None:
        args_cfg.seed = cli_args.seed
    if cli_args.epochs is not None:
        args_cfg.epochs = cli_args.epochs

    random_seed(args_cfg.seed)
    device = torch.device("cpu" if cli_args.cpu or not torch.cuda.is_available() else "cuda:0")

    inputs = load_inputs(project_root, cli_args.dataset)
    ncrna_feat = torch.from_numpy(inputs["ncrna_sim"]).float().to(device)
    drug_feat = torch.from_numpy(inputs["drug_sim"]).float().to(device)
    disease_feat = torch.from_numpy(inputs["disease_sim"]).float().to(device)
    all_pos = inputs["assoc"]

    dims = (ncrna_feat.shape[0], drug_feat.shape[0], disease_feat.shape[0])
    positive_set = {tuple(map(int, row[:3])) for row in all_pos}
    group_labels = all_pos[:, AXIS_TO_COL[cli_args.split_axis]]

    print(f"Dataset: {cli_args.dataset}")
    print(f"Split axis: {cli_args.split_axis}")
    print(f"Device: {device}")
    print(f"Seed: {args_cfg.seed}")
    print(f"Epochs: {args_cfg.epochs}")
    print(f"Patience: {cli_args.patience}")

    outer_cv = GroupKFold(n_splits=args_cfg.k_fold)
    fold_results = []

    for fold_num, (outer_train_idx, test_idx) in enumerate(outer_cv.split(all_pos, groups=group_labels), start=1):
        train_pool_pos = all_pos[outer_train_idx].astype(int)
        test_pos = all_pos[test_idx].astype(int)

        inner_train_pos, val_pos = split_train_val(
            train_pool_pos,
            split_axis=cli_args.split_axis,
            seed=args_cfg.seed + fold_num,
            val_ratio=cli_args.val_ratio,
        )

        train_axis_ids = np.unique(inner_train_pos[:, AXIS_TO_COL[cli_args.split_axis]])
        val_axis_ids = np.unique(val_pos[:, AXIS_TO_COL[cli_args.split_axis]])
        test_axis_ids = np.unique(test_pos[:, AXIS_TO_COL[cli_args.split_axis]])

        train_neg = sample_neutral_triplets(
            len(inner_train_pos), train_axis_ids, cli_args.split_axis, dims, positive_set, args_cfg.seed * 100 + fold_num * 10 + 1
        )
        val_neg = sample_neutral_triplets(
            len(val_pos), val_axis_ids, cli_args.split_axis, dims, positive_set, args_cfg.seed * 100 + fold_num * 10 + 2
        )
        test_neg = sample_neutral_triplets(
            len(test_pos), test_axis_ids, cli_args.split_axis, dims, positive_set, args_cfg.seed * 100 + fold_num * 10 + 3
        )

        train_data = np.vstack([inner_train_pos, train_neg]).astype(int)
        val_data = np.vstack([val_pos, val_neg]).astype(int)
        test_data = np.vstack([test_pos, test_neg]).astype(int)

        rng = np.random.default_rng(args_cfg.seed + fold_num)
        rng.shuffle(train_data)
        rng.shuffle(val_data)
        rng.shuffle(test_data)

        print(
            f"\n===== Fold {fold_num} =====\n"
            f"train_pos={len(inner_train_pos)}, train_neg={len(train_neg)}, total_train={len(train_data)}\n"
            f"val_pos={len(val_pos)}, val_neg={len(val_neg)}, total_val={len(val_data)}\n"
            f"test_pos={len(test_pos)}, test_neg={len(test_neg)}, total_test={len(test_data)}"
        )

        hyperedge_index, sign_weights = build_relational_hypergraph(
            train_data,
            ncrna_feat.shape[0],
            drug_feat.shape[0],
            disease_feat.shape[0],
            device,
        )

        model = create_model(args_cfg, ncrna_feat, drug_feat, disease_feat, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args_cfg.lr)
        loss_fn = FocalLoss(gamma=1, reduction="mean")

        best_metric = -1.0
        best_epoch = 0
        patience_counter = 0
        checkpoint_path = output_dir / f"best_model_fold_{fold_num}.pth"

        for epoch in range(args_cfg.epochs):
            total_loss, train_time, train_mem = train(
                ncrna_feat,
                drug_feat,
                disease_feat,
                train_data,
                model,
                optimizer,
                loss_fn,
                device,
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights,
            )
            val_metrics = test(
                ncrna_feat,
                drug_feat,
                disease_feat,
                val_data,
                model,
                device,
                fold_num,
                str(output_dir),
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights,
            )
            current_metric = float(val_metrics[cli_args.monitor_metric])
            print(
                f"Fold {fold_num} Epoch {epoch + 1}/{args_cfg.epochs} | "
                f"loss={total_loss:.6f} | train_time={train_time:.2f}s | mem={train_mem:.2f}GB | "
                f"val_f1={val_metrics['f1_macro']:.4f} | val_mcc={val_metrics['mcc']:.4f} | val_aupr={val_metrics['aupr_ovr']:.4f}"
            )

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                patience_counter = 0
                save_checkpoint(checkpoint_path, fold_num, epoch, model, optimizer, best_metric, cli_args.monitor_metric)
                print(f"  best checkpoint updated at epoch {best_epoch}: {cli_args.monitor_metric}={best_metric:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= cli_args.patience:
                    print(f"  early stopping triggered after {cli_args.patience} epochs without improvement")
                    break

        best_model = create_model(args_cfg, ncrna_feat, drug_feat, disease_feat, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        best_model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = test(
            ncrna_feat,
            drug_feat,
            disease_feat,
            test_data,
            best_model,
            device,
            fold_num,
            str(output_dir),
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights,
        )
        report = classification_report(
            test_metrics["labels"],
            test_metrics["preds"],
            labels=[0, 1, 2],
            target_names=["neutral", "resistance", "sensitivity"],
            zero_division=0,
        )
        print(
            f"Fold {fold_num} best_epoch={best_epoch} | test_acc={test_metrics['accuracy']:.4f} | "
            f"test_f1={test_metrics['f1_macro']:.4f} | test_mcc={test_metrics['mcc']:.4f} | test_aupr={test_metrics['aupr_ovr']:.4f}"
        )

        scalar_metrics = {}
        for key, value in test_metrics.items():
            if key in {"preds", "labels", "class_precision", "class_recall", "class_f1"}:
                continue
            scalar_metrics[key] = float(value) if isinstance(value, (int, float, np.floating)) else value

        fold_results.append(
            {
                "fold_num": fold_num,
                "best_epoch": best_epoch,
                "metrics": scalar_metrics,
                "report": report,
            }
        )

    summary = summarize_metrics([fold["metrics"] for fold in fold_results])
    write_summary(output_dir, fold_results, summary, cli_args.split_axis)

    print("\n===== Cold-start summary =====")
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc", "auc_ovr", "aupr_ovr"]:
        stats = summary[key]
        print(f"{key}: {stats['mean'] * 100:.2f} +- {stats['std'] * 100:.2f}")


if __name__ == "__main__":
    main()
