from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data" / "processed_pathway_only"
DEFAULT_CONTEXT_ROOT = ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "prior_noise_resources"
DEFAULT_CONTEXT_GROUPS = ("module_score", "spatial_proxy")
SEED = 48

RELATION_SPECS = {
    "rna_pathway": ("ncrna_id", "pathway_id"),
    "drug_pathway": ("drug_id", "pathway_id"),
    "disease_pathway": ("disease_id", "pathway_id"),
}


def _copy_tree(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            return
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _sample_size(total: int, fraction: float, minimum: int = 1) -> int:
    if total <= 0 or fraction <= 0.0:
        return 0
    size = int(round(total * fraction))
    return min(total, max(minimum, size))


def _load_selected_context_columns(context_root: Path, feature_groups: tuple[str, ...]) -> list[str]:
    manifest = pd.read_csv(context_root / "disease_aux_feature_manifest.csv")
    selected = [
        str(row.feature_name)
        for row in manifest.itertuples(index=False)
        if str(row.feature_group) in feature_groups
    ]
    if not selected:
        raise ValueError(f"No context features matched groups: {feature_groups}")
    return selected


def _random_derangement(indices: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    if len(indices) < 2:
        raise ValueError("A derangement requires at least two indices.")
    permuted = indices.copy()
    while True:
        rng.shuffle(permuted)
        if not np.any(permuted == indices):
            return permuted.copy()


def corrupt_pathway_relations(
    *,
    clean_data_root: Path,
    noisy_data_root: Path,
    noise_fraction: float,
    seed: int,
) -> dict[str, int]:
    rng = np.random.RandomState(seed)
    pathway_ids = pd.read_csv(clean_data_root / "nodes" / "pathway.csv")["id"].to_numpy(dtype=np.int64)
    relation_counts: dict[str, int] = {}

    for relation_name, (src_col, dst_col) in RELATION_SPECS.items():
        edge_path = noisy_data_root / "edges" / f"{relation_name}.csv"
        edge_df = pd.read_csv(edge_path)
        n_rows = len(edge_df)
        n_corrupt = _sample_size(n_rows, noise_fraction)
        relation_counts[relation_name] = int(n_corrupt)
        if n_corrupt == 0:
            continue

        chosen_positions = rng.choice(n_rows, size=n_corrupt, replace=False)
        adjacency: dict[int, set[int]] = {}
        for row in edge_df.itertuples(index=False):
            adjacency.setdefault(int(getattr(row, src_col)), set()).add(int(getattr(row, dst_col)))

        for row_pos in chosen_positions:
            src_id = int(edge_df.at[row_pos, src_col])
            old_dst = int(edge_df.at[row_pos, dst_col])
            blocked = adjacency[src_id] - {old_dst}
            candidates = np.setdiff1d(pathway_ids, np.fromiter(blocked | {old_dst}, dtype=np.int64), assume_unique=False)
            if len(candidates) == 0:
                candidates = np.setdiff1d(pathway_ids, np.array([old_dst], dtype=np.int64), assume_unique=False)
            if len(candidates) == 0:
                continue
            new_dst = int(rng.choice(candidates))
            adjacency[src_id].discard(old_dst)
            adjacency[src_id].add(new_dst)
            edge_df.at[row_pos, dst_col] = new_dst

        edge_df.to_csv(edge_path, index=False)

    return relation_counts


def corrupt_context_features(
    *,
    clean_context_root: Path,
    noisy_context_root: Path,
    noise_fraction: float,
    seed: int,
    feature_groups: tuple[str, ...],
) -> dict[str, object]:
    rng = np.random.RandomState(seed)
    selected_columns = _load_selected_context_columns(clean_context_root, feature_groups)
    aux_path = noisy_context_root / "disease_aux_features.csv"
    aux_df = pd.read_csv(aux_path)
    n_rows = len(aux_df)
    n_corrupt = _sample_size(n_rows, noise_fraction, minimum=2)
    if n_corrupt < 2:
        return {"n_corrupt_diseases": 0, "feature_columns": selected_columns, "permutation_applied": False}

    chosen_positions = np.sort(rng.choice(n_rows, size=n_corrupt, replace=False))
    permuted_positions = _random_derangement(chosen_positions.copy(), rng)

    selected_block = aux_df.loc[chosen_positions, selected_columns].to_numpy(dtype=np.float32)
    donor_block = aux_df.loc[permuted_positions, selected_columns].to_numpy(dtype=np.float32)
    aux_df.loc[chosen_positions, selected_columns] = donor_block
    aux_df.to_csv(aux_path, index=False)

    mapping_df = pd.DataFrame(
        {
            "target_row": chosen_positions,
            "donor_row": permuted_positions,
            "target_disease_id": aux_df.loc[chosen_positions, "disease_id"].to_numpy(dtype=np.int64),
            "donor_disease_id": aux_df.loc[permuted_positions, "disease_id"].to_numpy(dtype=np.int64),
        }
    )
    mapping_df.to_csv(noisy_context_root / "context_permutation_manifest.csv", index=False)

    return {
        "n_corrupt_diseases": int(n_corrupt),
        "feature_columns": selected_columns,
        "permutation_applied": True,
    }


def build_noisy_resource_pair(
    *,
    clean_data_root: Path,
    clean_context_root: Path,
    output_root: Path,
    scenario_name: str,
    pathway_noise_fraction: float,
    context_noise_fraction: float,
    feature_groups: tuple[str, ...] = DEFAULT_CONTEXT_GROUPS,
    overwrite: bool = False,
    seed: int = SEED,
) -> dict[str, object]:
    scenario_root = output_root / scenario_name
    noisy_data_root = scenario_root / "data_root"
    noisy_context_root = scenario_root / "context_root"
    scenario_root.mkdir(parents=True, exist_ok=True)

    _copy_tree(clean_data_root, noisy_data_root, overwrite=overwrite)
    _copy_tree(clean_context_root, noisy_context_root, overwrite=overwrite)

    pathway_counts = corrupt_pathway_relations(
        clean_data_root=clean_data_root,
        noisy_data_root=noisy_data_root,
        noise_fraction=pathway_noise_fraction,
        seed=seed,
    )
    context_summary = corrupt_context_features(
        clean_context_root=clean_context_root,
        noisy_context_root=noisy_context_root,
        noise_fraction=context_noise_fraction,
        seed=seed + 97,
        feature_groups=feature_groups,
    )

    metadata = {
        "scenario_name": scenario_name,
        "clean_data_root": str(clean_data_root),
        "clean_context_root": str(clean_context_root),
        "noisy_data_root": str(noisy_data_root),
        "noisy_context_root": str(noisy_context_root),
        "pathway_noise_fraction": float(pathway_noise_fraction),
        "context_noise_fraction": float(context_noise_fraction),
        "feature_groups": list(feature_groups),
        "pathway_corruption_counts": pathway_counts,
        "context_corruption_summary": context_summary,
        "seed": int(seed),
    }
    (scenario_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_feature_groups(raw_value: str) -> tuple[str, ...]:
    groups = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not groups:
        raise ValueError("At least one context feature group is required.")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--clean-context-root", type=str, default=str(DEFAULT_CONTEXT_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scenario-name", type=str, required=True)
    parser.add_argument("--pathway-noise", type=float, default=0.0)
    parser.add_argument("--context-noise", type=float, default=0.0)
    parser.add_argument("--feature-groups", type=str, default="module_score,spatial_proxy")
    parser.add_argument("--overwrite", choices=["yes", "no"], default="no")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    metadata = build_noisy_resource_pair(
        clean_data_root=Path(args.clean_data_root),
        clean_context_root=Path(args.clean_context_root),
        output_root=Path(args.output_root),
        scenario_name=args.scenario_name,
        pathway_noise_fraction=float(args.pathway_noise),
        context_noise_fraction=float(args.context_noise),
        feature_groups=parse_feature_groups(args.feature_groups),
        overwrite=(args.overwrite == "yes"),
        seed=int(args.seed),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
