from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_sphlcMga_core_vs_pathway.py"
DATA_ROOT = ROOT / "data" / "processed_pathway_only"
CONTEXT_AUX_ROOT = ROOT / "data" / "processed"
OUTPUT_ROOT = ROOT / "outputs_sphlcMga_pathway_context_study"


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=str(ROOT))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_row(label: str, metrics: dict) -> dict[str, str]:
    scores = metrics["best_validation_metrics"]
    score_std = metrics.get("best_validation_metrics_std", {})
    return {
        "Setting": label,
        "Accuracy": f"{scores['accuracy'] * 100:.2f}",
        "Accuracy Std": f"{score_std.get('accuracy', 0.0) * 100:.2f}",
        "Macro-F1": f"{scores['macro_f1'] * 100:.2f}",
        "Macro-F1 Std": f"{score_std.get('macro_f1', 0.0) * 100:.2f}",
        "AUC": f"{scores['auc_ovr'] * 100:.2f}",
        "AUC Std": f"{score_std.get('auc_ovr', 0.0) * 100:.2f}",
        "AUPR": f"{scores['aupr_ovr'] * 100:.2f}",
        "AUPR Std": f"{score_std.get('aupr_ovr', 0.0) * 100:.2f}",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--context-aux-root", type=str, default=str(CONTEXT_AUX_ROOT))
    parser.add_argument("--context-feature-groups", type=str, default="module_score,spatial_proxy")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--negative-sampling", choices=["fixed_pool", "baseline_dynamic"], default="baseline_dynamic")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--select-metric", choices=["macro_f1", "aupr_ovr", "auc_ovr", "accuracy"], default="macro_f1")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_dir = output_root / "base"
    pathway_dir = output_root / "pathway_all"
    context_dir = output_root / "context_aux"
    context_pathway_dir = output_root / "context_pathway_all"

    common_args = [
        "--data-root",
        args.data_root,
        "--context-aux-root",
        args.context_aux_root,
        "--context-feature-groups",
        args.context_feature_groups,
        "--cv-folds",
        str(args.cv_folds),
        "--negative-sampling",
        args.negative_sampling,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--gamma",
        str(args.gamma),
        "--select-metric",
        args.select_metric,
    ]

    run_command([
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(base_dir),
        "--model-type",
        "base",
        *common_args,
    ])
    run_command([
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(pathway_dir),
        "--model-type",
        "pathway",
        "--pathway-relations",
        "all",
        *common_args,
    ])
    run_command([
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(context_dir),
        "--model-type",
        "context",
        "--pathway-relations",
        "none",
        *common_args,
    ])
    run_command([
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(context_pathway_dir),
        "--model-type",
        "context_pathway",
        "--pathway-relations",
        "all",
        *common_args,
    ])

    rows = [
        extract_row("SPHLCMGA", load_metrics(base_dir / "metrics.json")),
        extract_row("SPHLCMGA + Pathway", load_metrics(pathway_dir / "metrics.json")),
        extract_row("SPHLCMGA + Context", load_metrics(context_dir / "metrics.json")),
        extract_row("SPHLCMGA + Context + Pathway", load_metrics(context_pathway_dir / "metrics.json")),
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
