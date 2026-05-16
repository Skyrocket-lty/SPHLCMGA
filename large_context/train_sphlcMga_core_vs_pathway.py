from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT.parent
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from model import NcrnaDrugDiseaseDualBranchModel, parameters_set
from dataset_utils import load_mapping_table, normalize_name
from sphlcMga_pathway_augmented_model import PathwayAugmentedSPHLCMGA, PathwayCounts

SEED = 48
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_ROOT = ROOT / "data" / "processed_pathway_only"
OUTPUT_ROOT = ROOT / "outputs_sphlcmga_pathway"
SOURCE_ROOT = WORK_ROOT

RELATION_SPECS = {
    "rna_pathway": ("ncrna_id", "pathway_id"),
    "drug_pathway": ("drug_id", "pathway_id"),
    "disease_pathway": ("disease_id", "pathway_id"),
}
VALID_CONTEXT_FEATURE_GROUPS = {"cell_composition", "module_score", "spatial_proxy"}


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 0.3, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        log_p_t = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        modulating = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, labels)
            loss = -alpha_t * modulating * log_p_t
        else:
            loss = -modulating * log_p_t
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def load_table(data_root: Path, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(data_root / relative_path)


def parse_context_feature_groups(raw_value: str) -> list[str]:
    groups = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [group for group in groups if group not in VALID_CONTEXT_FEATURE_GROUPS]
    if invalid:
        raise ValueError(f"Unsupported context feature groups: {invalid}")
    return groups


def load_aligned_context_aux_features(
    data_root: Path,
    context_aux_root: Path,
    feature_groups: list[str],
) -> tuple[torch.Tensor, list[str], str]:
    if not feature_groups:
        return torch.zeros((len(load_table(data_root, "nodes/disease.csv")), 0), dtype=torch.float32), [], "disabled"

    manifest_df = load_table(context_aux_root, "disease_aux_feature_manifest.csv")
    selected_columns = [
        str(row.feature_name)
        for row in manifest_df.itertuples(index=False)
        if str(row.feature_group) in feature_groups
    ]
    if not selected_columns:
        raise ValueError(f"No context auxiliary columns selected for groups: {feature_groups}")

    target_disease_df = load_table(data_root, "nodes/disease.csv").sort_values("id").reset_index(drop=True)
    source_disease_df = load_table(context_aux_root, "nodes/disease.csv").sort_values("id").reset_index(drop=True)
    aux_df = load_table(context_aux_root, "disease_aux_features.csv")
    source_feature_df = aux_df.merge(
        source_disease_df[["id", "name", "canonical_name"]],
        left_on="disease_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    )
    if source_feature_df["canonical_name"].isna().any():
        raise ValueError("Context auxiliary features could not be aligned to source disease nodes.")

    source_feature_df["lookup_key"] = source_feature_df["canonical_name"].fillna(source_feature_df["name"]).map(normalize_name)
    if source_feature_df["lookup_key"].duplicated().any():
        duplicates = source_feature_df.loc[source_feature_df["lookup_key"].duplicated(), "lookup_key"].tolist()
        raise ValueError(f"Duplicate context auxiliary disease keys detected: {duplicates[:5]}")

    target_lookup = target_disease_df["canonical_name"].fillna(target_disease_df["name"]).map(normalize_name)
    source_lookup = source_disease_df["canonical_name"].fillna(source_disease_df["name"]).map(normalize_name)
    direct_match = (
        len(target_disease_df) == len(source_disease_df)
        and np.array_equal(target_disease_df["id"].to_numpy(), source_disease_df["id"].to_numpy())
        and np.array_equal(target_lookup.to_numpy(), source_lookup.to_numpy())
    )

    if direct_match:
        aligned_df = source_feature_df.sort_values("disease_id").reset_index(drop=True)
        alignment_mode = "direct_id_order"
    else:
        feature_by_key = source_feature_df.set_index("lookup_key")
        aligned_rows = []
        missing_keys: list[str] = []
        for row in target_disease_df.itertuples(index=False):
            lookup_key = normalize_name(str(row.canonical_name) if pd.notna(row.canonical_name) else str(row.name))
            if lookup_key not in feature_by_key.index:
                missing_keys.append(lookup_key)
                continue
            aligned_rows.append(feature_by_key.loc[lookup_key, selected_columns].to_numpy(dtype=np.float32))
        if missing_keys:
            raise ValueError(f"Missing context auxiliary features for diseases: {missing_keys[:5]}")
        aligned_df = pd.DataFrame(aligned_rows, columns=selected_columns)
        alignment_mode = "canonical_name_reindexed"

    features = torch.tensor(aligned_df[selected_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    return features, selected_columns, alignment_mode


def build_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    probs = np.nan_to_num(probs, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 1e-8, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(1) <= 0
    if np.any(zero_rows):
        probs[zero_rows] = 1.0 / 3.0
        row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    preds = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "resistance_f1": float(f1_score(y_true, preds, labels=[1], average="macro", zero_division=0)),
    }
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    metrics["auc_ovr"] = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro"))
    metrics["aupr_ovr"] = float(average_precision_score(y_bin, probs, average="macro"))
    return metrics


def build_relational_hypergraph(
    train_df: pd.DataFrame,
    n_ncrna: int,
    n_drug: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_df = train_df[train_df["label"].isin([1, 2])].reset_index(drop=True)
    node_indices = []
    edge_indices = []
    sign_weights = []
    drug_offset = n_ncrna
    disease_offset = n_ncrna + n_drug
    for edge_id, row in enumerate(positive_df.itertuples(index=False)):
        node_indices.extend(
            [
                int(row.ncrna_id),
                int(row.drug_id) + drug_offset,
                int(row.disease_id) + disease_offset,
            ]
        )
        edge_indices.extend([edge_id, edge_id, edge_id])
        sign_weights.append(-1.0 if int(row.label) == 1 else 1.0)
    if not sign_weights:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
        )
    hyperedge_index = torch.tensor([node_indices, edge_indices], dtype=torch.long, device=device)
    sign_tensor = torch.tensor(sign_weights, dtype=torch.float32, device=device)
    return hyperedge_index, sign_tensor


def build_union_feature_matrix(
    node_type: str,
    union_nodes: pd.DataFrame,
    dataset_dir: Path,
    mapping_file: Path,
    base_file: str,
    llm_file: str,
) -> tuple[np.ndarray, np.ndarray]:
    _, local_map = load_mapping_table(mapping_file, dataset_dir.name)
    local_lookup = local_map[node_type]
    union_lookup = {
        normalize_name(row.name): int(row.id)
        for row in union_nodes.itertuples(index=False)
    }
    local_to_union = {
        int(local_id): union_lookup[normalize_name(name)]
        for local_id, name in local_lookup.items()
        if normalize_name(name) in union_lookup
    }
    base_sim = np.loadtxt(dataset_dir / base_file)
    llm_sim = np.loadtxt(dataset_dir / llm_file)
    local_sim = (base_sim + llm_sim) / 2.0

    union_size = len(union_nodes)
    sum_mat = np.zeros((union_size, union_size), dtype=np.float32)
    count_mat = np.zeros((union_size, union_size), dtype=np.float32)
    local_ids = sorted(local_to_union.keys())
    for i in local_ids:
        ui = local_to_union[i]
        for j in local_ids:
            uj = local_to_union[j]
            sum_mat[ui, uj] += float(local_sim[i, j])
            count_mat[ui, uj] += 1.0
    return sum_mat.astype(np.float32), count_mat.astype(np.float32)


def build_combined_features(data_root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ncrna_nodes = load_table(data_root, "nodes/ncrna.csv").sort_values("id").reset_index(drop=True)
    drug_nodes = load_table(data_root, "nodes/drug.csv").sort_values("id").reset_index(drop=True)
    disease_nodes = load_table(data_root, "nodes/disease.csv").sort_values("id").reset_index(drop=True)

    mapping1 = SOURCE_ROOT / "datasets" / "metadata" / "entity_mapping-dataset1.csv"
    mapping2 = SOURCE_ROOT / "datasets" / "metadata" / "entity_mapping-dataset2.csv"

    ncrna_sum1, ncrna_cnt1 = build_union_feature_matrix("ncrna", ncrna_nodes, SOURCE_ROOT / "datasets" / "dataset1", mapping1, "ncRNA.txt", "LLM_rna_sim.txt")
    ncrna_sum2, ncrna_cnt2 = build_union_feature_matrix("ncrna", ncrna_nodes, SOURCE_ROOT / "datasets" / "dataset2", mapping2, "RNA.txt", "LLM_rna_sim.txt")
    drug_sum1, drug_cnt1 = build_union_feature_matrix("drug", drug_nodes, SOURCE_ROOT / "datasets" / "dataset1", mapping1, "drug.txt", "LLM_drug_sim.txt")
    drug_sum2, drug_cnt2 = build_union_feature_matrix("drug", drug_nodes, SOURCE_ROOT / "datasets" / "dataset2", mapping2, "drug.txt", "LLM_drug_sim.txt")
    disease_sum1, disease_cnt1 = build_union_feature_matrix("disease", disease_nodes, SOURCE_ROOT / "datasets" / "dataset1", mapping1, "disease.txt", "LLM_disease_sim.txt")
    disease_sum2, disease_cnt2 = build_union_feature_matrix("disease", disease_nodes, SOURCE_ROOT / "datasets" / "dataset2", mapping2, "disease.txt", "LLM_disease_sim.txt")

    ncrna_feat = np.divide(ncrna_sum1 + ncrna_sum2, np.maximum(ncrna_cnt1 + ncrna_cnt2, 1.0))
    drug_feat = np.divide(drug_sum1 + drug_sum2, np.maximum(drug_cnt1 + drug_cnt2, 1.0))
    disease_feat = np.divide(disease_sum1 + disease_sum2, np.maximum(disease_cnt1 + disease_cnt2, 1.0))

    np.fill_diagonal(ncrna_feat, 1.0)
    np.fill_diagonal(drug_feat, 1.0)
    np.fill_diagonal(disease_feat, 1.0)

    return (
        torch.tensor(ncrna_feat, dtype=torch.float32),
        torch.tensor(drug_feat, dtype=torch.float32),
        torch.tensor(disease_feat, dtype=torch.float32),
    )


def build_pathway_relations(
    data_root: Path,
    device: torch.device,
    relation_mode: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, int]]:
    relation_aliases = {
        "none": set(),
        "all": {"rna_pathway", "drug_pathway", "disease_pathway"},
        "rna_only": {"rna_pathway"},
        "drug_only": {"drug_pathway"},
        "disease_only": {"disease_pathway"},
    }
    enabled = relation_aliases[relation_mode]
    relations: dict[str, dict[str, torch.Tensor]] = {}
    relation_edge_counts: dict[str, int] = {}
    for edge_name, (left_key, right_key) in RELATION_SPECS.items():
        if edge_name in enabled:
            edge_df = load_table(data_root, f"edges/{edge_name}.csv")
        else:
            edge_df = pd.DataFrame(columns=[left_key, right_key, "weight"])
        if edge_df.empty:
            left_index = torch.empty(0, dtype=torch.long, device=device)
            right_index = torch.empty(0, dtype=torch.long, device=device)
            weight = torch.empty(0, dtype=torch.float32, device=device)
        else:
            left_index = torch.tensor(edge_df[left_key].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
            right_index = torch.tensor(edge_df[right_key].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
            weight = torch.tensor(edge_df["weight"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        relations[edge_name] = {
            f"{left_key.replace('_id', '')}_index": left_index,
            f"{right_key.replace('_id', '')}_index": right_index,
            "weight": weight,
        }
        relation_edge_counts[edge_name] = int(len(edge_df))
    return relations, relation_edge_counts


def instantiate_model(
    model_type: str,
    data_root: Path,
    device: torch.device,
    ncrna_feat: torch.Tensor,
    drug_feat: torch.Tensor,
    disease_feat: torch.Tensor,
    relation_mode: str,
    disease_aux_features: torch.Tensor | None = None,
):
    args = parameters_set()
    if model_type == "base":
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
        return model, {}, {"rna_pathway": 0, "drug_pathway": 0, "disease_pathway": 0}

    use_pathway_branch = model_type in {"pathway", "context_pathway"}
    use_context_branch = model_type in {"context", "context_pathway"}
    effective_relation_mode = relation_mode if use_pathway_branch else "none"
    relations, relation_edge_counts = build_pathway_relations(data_root, device, effective_relation_mode)
    counts = PathwayCounts(
        ncrna=ncrna_feat.shape[0],
        drug=drug_feat.shape[0],
        disease=disease_feat.shape[0],
        pathway=len(load_table(data_root, "nodes/pathway.csv")),
    )
    model = PathwayAugmentedSPHLCMGA(
        counts=counts,
        relations=relations,
        ncrna_dim=ncrna_feat.shape[1],
        drug_dim=drug_feat.shape[1],
        disease_dim=disease_feat.shape[1],
        modal_out_dim=args.modal_out_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        hgnn_dim_1=args.hgnn_dim_1,
        disease_aux_features=disease_aux_features,
        use_pathway_branch=use_pathway_branch,
        use_context_branch=use_context_branch,
        dropout=args.dropout,
    ).to(device)
    return model, relations, relation_edge_counts


def tensorize_frame(frame: pd.DataFrame) -> TensorDataset:
    return TensorDataset(
        torch.tensor(frame["ncrna_id"].to_numpy(dtype=np.int64)),
        torch.tensor(frame["drug_id"].to_numpy(dtype=np.int64)),
        torch.tensor(frame["disease_id"].to_numpy(dtype=np.int64)),
        torch.tensor(frame["label"].to_numpy(dtype=np.int64)),
    )


def sample_baseline_non_association_splits(
    *,
    all_positive_df: pd.DataFrame,
    train_positive_df: pd.DataFrame,
    val_positive_df: pd.DataFrame,
    n_ncrna: int,
    n_drug: int,
    n_disease: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    positive_keys = {
        (int(row.ncrna_id), int(row.drug_id), int(row.disease_id))
        for row in all_positive_df.itertuples(index=False)
    }
    train_negative_keys: set[tuple[int, int, int]] = set()

    def draw_negative(forbid_train_duplicates: bool) -> tuple[int, int, int]:
        for _ in range(200000):
            candidate = (
                int(rng.randint(0, n_ncrna)),
                int(rng.randint(0, n_drug)),
                int(rng.randint(0, n_disease)),
            )
            if candidate in positive_keys:
                continue
            if candidate in train_negative_keys:
                continue
            if forbid_train_duplicates:
                train_negative_keys.add(candidate)
            return candidate
        raise RuntimeError("Unable to sample a baseline-style non-association triplet.")

    def build_negative_frame(source_df: pd.DataFrame, split_name: str, forbid_train_duplicates: bool) -> pd.DataFrame:
        rows = []
        for _ in source_df.itertuples(index=False):
            ncrna_id, drug_id, disease_id = draw_negative(forbid_train_duplicates=forbid_train_duplicates)
            rows.append(
                {
                    "ncrna_id": ncrna_id,
                    "drug_id": drug_id,
                    "disease_id": disease_id,
                    "label": 0,
                    "source_dataset": "dynamic_baseline_sampling",
                    "split_role": f"{split_name}_non_association",
                }
            )
        if not rows:
            return pd.DataFrame(columns=["ncrna_id", "drug_id", "disease_id", "label", "source_dataset", "split_role"])
        return pd.DataFrame(rows)

    train_neg_df = build_negative_frame(train_positive_df, "train", forbid_train_duplicates=True)
    val_neg_df = build_negative_frame(val_positive_df, "validation", forbid_train_duplicates=False)
    train_df = pd.concat([train_positive_df.copy(), train_neg_df], ignore_index=True)
    val_df = pd.concat([val_positive_df.copy(), val_neg_df], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def prepare_split_frames(
    *,
    all_positive_df: pd.DataFrame,
    train_positive_df: pd.DataFrame,
    val_positive_df: pd.DataFrame,
    args_cli: argparse.Namespace,
    n_ncrna: int,
    n_drug: int,
    n_disease: int,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args_cli.negative_sampling == "fixed_pool":
        return train_positive_df.copy().reset_index(drop=True), val_positive_df.copy().reset_index(drop=True)
    return sample_baseline_non_association_splits(
        all_positive_df=all_positive_df,
        train_positive_df=train_positive_df,
        val_positive_df=val_positive_df,
        n_ncrna=n_ncrna,
        n_drug=n_drug,
        n_disease=n_disease,
        seed=split_seed,
    )


def run_fold(
    *,
    fold_tag: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args_cli: argparse.Namespace,
    data_root: Path,
    device: torch.device,
    ncrna_feat: torch.Tensor,
    drug_feat: torch.Tensor,
    disease_feat: torch.Tensor,
    disease_aux_features: torch.Tensor | None,
    output_dir: Path,
) -> dict[str, object]:
    train_loader = DataLoader(tensorize_frame(train_df), batch_size=args_cli.batch_size, shuffle=True)
    val_loader = DataLoader(tensorize_frame(val_df), batch_size=768, shuffle=False)

    hyperedge_index, sign_weights = build_relational_hypergraph(
        train_df,
        n_ncrna=ncrna_feat.shape[0],
        n_drug=drug_feat.shape[0],
        device=device,
    )

    model, _, relation_edge_counts = instantiate_model(
        model_type=args_cli.model_type,
        data_root=data_root,
        device=device,
        ncrna_feat=ncrna_feat,
        drug_feat=drug_feat,
        disease_feat=disease_feat,
        relation_mode=args_cli.pathway_relations,
        disease_aux_features=disease_aux_features,
    )

    class_freq = train_df["label"].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / (3.0 * class_freq.get(label, 1)) for label in [0, 1, 2]],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = FocalLoss(alpha=class_weights, gamma=args_cli.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr, weight_decay=args_cli.weight_decay)

    best_state = None
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args_cli.epochs + 1):
        model.train()
        running_loss = 0.0
        skipped_batches = 0
        for batch in train_loader:
            ncrna_ids, drug_ids, disease_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad()
            logits, _, _ = model(
                ncrna_feat,
                drug_feat,
                disease_feat,
                ncrna_ids,
                drug_ids,
                disease_ids,
                hyperedge_index=hyperedge_index,
                sign_weights=sign_weights,
            )
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)

        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                ncrna_ids, drug_ids, disease_ids, labels = [item.to(device) for item in batch]
                logits, _, _ = model(
                    ncrna_feat,
                    drug_feat,
                    disease_feat,
                    ncrna_ids,
                    drug_ids,
                    disease_ids,
                    hyperedge_index=hyperedge_index,
                    sign_weights=sign_weights,
                )
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                pred_probs = torch.softmax(logits, dim=1)
                val_probs.append(pred_probs.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        val_probs_np = np.concatenate(val_probs, axis=0)
        val_labels_np = np.concatenate(val_labels, axis=0)
        metrics = build_metrics(val_labels_np, val_probs_np)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / len(train_df),
                **metrics,
            }
        )
        current_score = metrics[args_cli.select_metric]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        print(
            f"[{fold_tag}] Epoch {epoch:02d} | loss={history[-1]['train_loss']:.4f} | "
            f"acc={metrics['accuracy']:.4f} | f1={metrics['macro_f1']:.4f} | "
            f"auc={metrics['auc_ovr']:.4f} | aupr={metrics['aupr_ovr']:.4f} | "
            f"skipped={skipped_batches}"
        )
        if epochs_without_improvement >= args_cli.patience:
            print(f"[{fold_tag}] Early stopping at epoch {epoch:02d}; best epoch was {best_epoch:02d}.")
            break

    if best_state is None:
        raise RuntimeError(f"{fold_tag}: training did not produce a valid checkpoint.")

    torch.save(best_state, output_dir / f"{args_cli.model_type}_model.pt")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_dataset = tensorize_frame(val_df)
        logits, _, _ = model(
            ncrna_feat,
            drug_feat,
            disease_feat,
            val_dataset.tensors[0].to(device),
            val_dataset.tensors[1].to(device),
            val_dataset.tensors[2].to(device),
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights,
        )
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    val_metrics = build_metrics(val_df["label"].to_numpy(dtype=np.int64), probs)

    pred_df = val_df.copy().reset_index(drop=True)
    pred_df["pred_label"] = probs.argmax(axis=1)
    pred_df["prob_non_association"] = probs[:, 0]
    pred_df["prob_resistance"] = probs[:, 1]
    pred_df["prob_sensitivity"] = probs[:, 2]
    pred_df["confidence"] = probs.max(axis=1)
    pred_df.sort_values("confidence", ascending=False).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "validation_split.csv", index=False)

    fold_summary = {
        "fold_tag": fold_tag,
        "best_epoch": int(best_epoch),
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "train_hypergraph_positive_edges": int((train_df["label"] != 0).sum()),
        "relation_edge_counts": relation_edge_counts,
        "best_validation_metrics": val_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
    return fold_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--model-type", choices=["base", "pathway", "context", "context_pathway"], default="base")
    parser.add_argument("--pathway-relations", choices=["none", "all", "rna_only", "drug_only", "disease_only"], default="all")
    parser.add_argument("--context-aux-root", type=str, default=str(ROOT / "data" / "processed"))
    parser.add_argument("--context-feature-groups", type=str, default="module_score,spatial_proxy")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--select-metric", choices=["macro_f1", "aupr_ovr", "auc_ovr", "accuracy"], default="macro_f1")
    parser.add_argument("--cv-folds", type=int, default=1)
    parser.add_argument("--negative-sampling", choices=["fixed_pool", "baseline_dynamic"], default="fixed_pool")
    args_cli = parser.parse_args()

    data_root = Path(args_cli.data_root)
    output_root = Path(args_cli.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args_cli.negative_sampling == "fixed_pool":
        triplets = load_table(data_root, "triplets_full.csv")
        split_source = triplets
    else:
        triplets = load_table(data_root, "triplets_positive.csv")
        split_source = triplets
    ncrna_feat, drug_feat, disease_feat = build_combined_features(data_root)
    ncrna_feat = ncrna_feat.to(device)
    drug_feat = drug_feat.to(device)
    disease_feat = disease_feat.to(device)
    context_feature_groups = parse_context_feature_groups(args_cli.context_feature_groups)
    context_feature_columns: list[str] = []
    context_alignment_mode = "unused"
    disease_aux_features: torch.Tensor | None = None
    if args_cli.model_type in {"context", "context_pathway"}:
        disease_aux_features, context_feature_columns, context_alignment_mode = load_aligned_context_aux_features(
            data_root=data_root,
            context_aux_root=Path(args_cli.context_aux_root),
            feature_groups=context_feature_groups,
        )
        disease_aux_features = disease_aux_features.to(device)

    if args_cli.cv_folds <= 1:
        train_positive_df, val_positive_df = train_test_split(
            split_source,
            test_size=0.2,
            random_state=SEED,
            stratify=split_source["label"],
        )
        train_df, val_df = prepare_split_frames(
            all_positive_df=triplets,
            train_positive_df=train_positive_df,
            val_positive_df=val_positive_df,
            args_cli=args_cli,
            n_ncrna=ncrna_feat.shape[0],
            n_drug=drug_feat.shape[0],
            n_disease=disease_feat.shape[0],
            split_seed=SEED,
        )
        fold_summary = run_fold(
            fold_tag="split",
            train_df=train_df,
            val_df=val_df,
            args_cli=args_cli,
            data_root=data_root,
            device=device,
            ncrna_feat=ncrna_feat,
            drug_feat=drug_feat,
            disease_feat=disease_feat,
            disease_aux_features=disease_aux_features,
            output_dir=output_root,
        )
        summary = {
            "seed": SEED,
            "device": str(device),
            "model_type": args_cli.model_type,
            "pathway_relations": args_cli.pathway_relations if args_cli.model_type in {"pathway", "context_pathway"} else "none",
            "context_aux_root": str(Path(args_cli.context_aux_root)),
            "context_feature_groups": context_feature_groups,
            "context_feature_columns": context_feature_columns,
            "context_alignment_mode": context_alignment_mode,
            "epochs": int(args_cli.epochs),
            "best_epoch": int(fold_summary["best_epoch"]),
            "train_size": int(fold_summary["train_size"]),
            "validation_size": int(fold_summary["validation_size"]),
            "lr": float(args_cli.lr),
            "batch_size": int(args_cli.batch_size),
            "weight_decay": float(args_cli.weight_decay),
            "gamma": float(args_cli.gamma),
            "select_metric": args_cli.select_metric,
            "negative_sampling": args_cli.negative_sampling,
            "train_hypergraph_positive_edges": int(fold_summary["train_hypergraph_positive_edges"]),
            "relation_edge_counts": fold_summary["relation_edge_counts"],
            "best_validation_metrics": fold_summary["best_validation_metrics"],
        }
    else:
        splitter = StratifiedKFold(n_splits=args_cli.cv_folds, shuffle=True, random_state=SEED)
        fold_summaries = []
        for fold_index, (train_idx, val_idx) in enumerate(
            splitter.split(split_source, split_source["label"].to_numpy(dtype=np.int64)),
            start=1,
        ):
            fold_dir = output_root / f"fold_{fold_index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_positive_df = split_source.iloc[train_idx].copy().reset_index(drop=True)
            val_positive_df = split_source.iloc[val_idx].copy().reset_index(drop=True)
            train_df, val_df = prepare_split_frames(
                all_positive_df=triplets,
                train_positive_df=train_positive_df,
                val_positive_df=val_positive_df,
                args_cli=args_cli,
                n_ncrna=ncrna_feat.shape[0],
                n_drug=drug_feat.shape[0],
                n_disease=disease_feat.shape[0],
                split_seed=SEED + fold_index,
            )
            fold_summaries.append(
                run_fold(
                    fold_tag=f"fold_{fold_index}",
                    train_df=train_df,
                    val_df=val_df,
                    args_cli=args_cli,
                    data_root=data_root,
                    device=device,
                    ncrna_feat=ncrna_feat,
                    drug_feat=drug_feat,
                    disease_feat=disease_feat,
                    disease_aux_features=disease_aux_features,
                    output_dir=fold_dir,
                )
            )

        metric_names = list(fold_summaries[0]["best_validation_metrics"].keys())
        mean_metrics = {
            metric_name: float(
                np.mean([fold["best_validation_metrics"][metric_name] for fold in fold_summaries])
            )
            for metric_name in metric_names
        }
        std_metrics = {
            metric_name: float(
                np.std([fold["best_validation_metrics"][metric_name] for fold in fold_summaries])
            )
            for metric_name in metric_names
        }
        fold_rows = []
        for fold_index, fold_summary in enumerate(fold_summaries, start=1):
            row = {
                "fold": fold_index,
                "best_epoch": int(fold_summary["best_epoch"]),
                "train_size": int(fold_summary["train_size"]),
                "validation_size": int(fold_summary["validation_size"]),
            }
            row.update(fold_summary["best_validation_metrics"])
            fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(output_root / "cv_fold_metrics.csv", index=False)

        summary = {
            "seed": SEED,
            "device": str(device),
            "model_type": args_cli.model_type,
            "pathway_relations": args_cli.pathway_relations if args_cli.model_type in {"pathway", "context_pathway"} else "none",
            "context_aux_root": str(Path(args_cli.context_aux_root)),
            "context_feature_groups": context_feature_groups,
            "context_feature_columns": context_feature_columns,
            "context_alignment_mode": context_alignment_mode,
            "epochs": int(args_cli.epochs),
            "cv_folds": int(args_cli.cv_folds),
            "lr": float(args_cli.lr),
            "batch_size": int(args_cli.batch_size),
            "weight_decay": float(args_cli.weight_decay),
            "gamma": float(args_cli.gamma),
            "select_metric": args_cli.select_metric,
            "negative_sampling": args_cli.negative_sampling,
            "relation_edge_counts": fold_summaries[0]["relation_edge_counts"],
            "fold_best_epochs": [int(fold["best_epoch"]) for fold in fold_summaries],
            "fold_metrics": fold_rows,
            "best_validation_metrics": mean_metrics,
            "best_validation_metrics_std": std_metrics,
        }
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
