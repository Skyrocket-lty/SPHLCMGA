from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from build_noisy_prior_resources import DEFAULT_CONTEXT_GROUPS, build_noisy_resource_pair, parse_feature_groups

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_sphlcMga_core_vs_pathway.py"
BUILD_OUTPUT_ROOT = ROOT / "data" / "prior_noise_resources"
RUN_OUTPUT_ROOT = ROOT / "outputs_prior_noise_robustness"
ANALYSIS_SCRIPT = ROOT / "analyze_prior_noise_robustness.py"

DEFAULT_SCENARIOS = {
    "clean": {"pathway_noise": 0.0, "context_noise": 0.0},
    "pathway10": {"pathway_noise": 0.10, "context_noise": 0.0},
    "pathway20": {"pathway_noise": 0.20, "context_noise": 0.0},
    "combined10": {"pathway_noise": 0.10, "context_noise": 0.10},
    "combined20": {"pathway_noise": 0.20, "context_noise": 0.20},
}


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=str(ROOT))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: float) -> str:
    return f"{value * 100.0:.2f}"


def summary_row(scenario: str, label: str, metrics: dict) -> dict[str, str]:
    best = metrics["best_validation_metrics"]
    return {
        "scenario": scenario,
        "setting": label,
        "Accuracy": format_metric(float(best["accuracy"])),
        "Macro-F1": format_metric(float(best["macro_f1"])),
        "AUC": format_metric(float(best["auc_ovr"])),
        "AUPR": format_metric(float(best["aupr_ovr"])),
        "Resistance-F1": format_metric(float(best["resistance_f1"])),
    }


def scenario_roots(
    *,
    scenario_name: str,
    clean_data_root: Path,
    clean_context_root: Path,
    resource_root: Path,
    feature_groups: tuple[str, ...],
    overwrite_resources: bool,
) -> tuple[Path, Path]:
    if scenario_name == "clean":
        return clean_data_root, clean_context_root

    config = DEFAULT_SCENARIOS[scenario_name]
    metadata = build_noisy_resource_pair(
        clean_data_root=clean_data_root,
        clean_context_root=clean_context_root,
        output_root=resource_root,
        scenario_name=scenario_name,
        pathway_noise_fraction=float(config["pathway_noise"]),
        context_noise_fraction=float(config["context_noise"]),
        feature_groups=feature_groups,
        overwrite=overwrite_resources,
    )
    return Path(metadata["noisy_data_root"]), Path(metadata["noisy_context_root"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-data-root", type=str, default=str(ROOT / "data" / "processed_pathway_only"))
    parser.add_argument("--clean-context-root", type=str, default=str(ROOT / "data" / "processed"))
    parser.add_argument("--resource-root", type=str, default=str(BUILD_OUTPUT_ROOT))
    parser.add_argument("--output-root", type=str, default=str(RUN_OUTPUT_ROOT))
    parser.add_argument("--feature-groups", type=str, default="module_score,spatial_proxy")
    parser.add_argument("--scenarios", type=str, default="clean,pathway10,pathway20,combined10,combined20")
    parser.add_argument("--overwrite-resources", choices=["yes", "no"], default="no")
    parser.add_argument("--overwrite-runs", choices=["yes", "no"], default="no")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--select-metric", choices=["macro_f1", "aupr_ovr", "auc_ovr", "accuracy"], default="macro_f1")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--negative-sampling", choices=["fixed_pool", "baseline_dynamic"], default="baseline_dynamic")
    parser.add_argument("--run-analysis", choices=["yes", "no"], default="yes")
    args = parser.parse_args()

    clean_data_root = Path(args.clean_data_root)
    clean_context_root = Path(args.clean_context_root)
    resource_root = Path(args.resource_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_groups = parse_feature_groups(args.feature_groups)
    scenario_names = [item.strip() for item in args.scenarios.split(",") if item.strip()]

    rows: list[dict[str, str]] = []
    common_args = [
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
        "--cv-folds",
        str(args.cv_folds),
        "--negative-sampling",
        args.negative_sampling,
        "--context-feature-groups",
        args.feature_groups,
    ]

    for scenario_name in scenario_names:
        if scenario_name not in DEFAULT_SCENARIOS:
            raise ValueError(f"Unsupported scenario: {scenario_name}")

        data_root, context_root = scenario_roots(
            scenario_name=scenario_name,
            clean_data_root=clean_data_root,
            clean_context_root=clean_context_root,
            resource_root=resource_root,
            feature_groups=feature_groups,
            overwrite_resources=(args.overwrite_resources == "yes"),
        )

        scenario_output = output_root / scenario_name
        scenario_output.mkdir(parents=True, exist_ok=True)
        for model_type, label in [("base", "SPHLCMGA"), ("context_pathway", "SPHLCMGA-CP")]:
            run_dir = scenario_output / model_type
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists() and args.overwrite_runs == "no":
                metrics = load_metrics(metrics_path)
                rows.append(summary_row(scenario_name, label, metrics))
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--data-root",
                str(data_root),
                "--context-aux-root",
                str(context_root),
                "--output-dir",
                str(run_dir),
                "--model-type",
                model_type,
                "--pathway-relations",
                "all" if model_type == "context_pathway" else "none",
                *common_args,
            ]
            run_command(command)
            metrics = load_metrics(metrics_path)
            rows.append(summary_row(scenario_name, label, metrics))

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "clean_data_root": str(clean_data_root),
        "clean_context_root": str(clean_context_root),
        "resource_root": str(resource_root),
        "output_root": str(output_root),
        "feature_groups": list(feature_groups),
        "scenarios": scenario_names,
        "model_settings": ["base", "context_pathway"],
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "gamma": float(args.gamma),
        "select_metric": args.select_metric,
        "cv_folds": int(args.cv_folds),
        "negative_sampling": args.negative_sampling,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.run_analysis == "yes":
        analysis_output = output_root / "analysis"
        run_command(
            [
                sys.executable,
                str(ANALYSIS_SCRIPT),
                "--runs-root",
                str(output_root),
                "--output-dir",
                str(analysis_output),
                "--scenario-order",
                ",".join(scenario_names),
            ]
        )


if __name__ == "__main__":
    main()
