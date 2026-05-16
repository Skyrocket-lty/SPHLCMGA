from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyze_differential_gain import align_predictions, load_predictions_dir

ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = ROOT / "outputs_prior_noise_robustness"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "analysis"

MODEL_LABELS = {
    "base": "SPHLCMGA",
    "context_pathway": "SPHLCMGA-CP",
}

SCENARIO_LABELS = {
    "clean": "Clean",
    "pathway10": "Pathway 10%",
    "pathway20": "Pathway 20%",
    "combined10": "Combined 10%",
    "combined20": "Combined 20%",
}


def scenario_family(name: str) -> str:
    if name == "clean":
        return "clean"
    if name.startswith("pathway"):
        return "pathway"
    if name.startswith("combined"):
        return "combined"
    raise ValueError(f"Unsupported scenario name: {name}")


def scenario_level(name: str) -> int:
    if name == "clean":
        return 0
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits)


def load_prediction_bundle(result_dir: Path) -> dict[str, pd.DataFrame]:
    fold_dirs = sorted([path for path in result_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")])
    if fold_dirs:
        return load_predictions_dir(result_dir)
    single_path = result_dir / "validation_predictions.csv"
    if not single_path.exists():
        raise FileNotFoundError(f"Missing validation predictions under {result_dir}")
    frame = pd.read_csv(single_path)
    frame["fold"] = "split"
    return {"split": frame}


def load_metrics_rows(runs_root: Path, scenario_names: list[str]) -> pd.DataFrame:
    rows = []
    for scenario_name in scenario_names:
        for model_name, label in MODEL_LABELS.items():
            metrics_path = runs_root / scenario_name / model_name / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            best = metrics["best_validation_metrics"]
            rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_label": SCENARIO_LABELS.get(scenario_name, scenario_name),
                    "scenario_family": scenario_family(scenario_name),
                    "noise_level": scenario_level(scenario_name),
                    "model": model_name,
                    "model_label": label,
                    "accuracy": float(best["accuracy"]),
                    "macro_f1": float(best["macro_f1"]),
                    "auc_ovr": float(best["auc_ovr"]),
                    "aupr_ovr": float(best["aupr_ovr"]),
                    "resistance_f1": float(best["resistance_f1"]),
                }
            )
    return pd.DataFrame(rows)


def compute_polarity_flip_rate(clean_dir: Path, noisy_dir: Path) -> dict[str, float]:
    clean_frames = load_prediction_bundle(clean_dir)
    noisy_frames = load_prediction_bundle(noisy_dir)
    aligned = align_predictions(clean_frames, noisy_frames)
    clean_positive_correct = aligned["label"].isin([1, 2]) & (aligned["pred_label_standard"] == aligned["label"])
    flip_to_opposite = clean_positive_correct & (aligned["pred_label_rich"].isin([1, 2])) & (
        aligned["pred_label_rich"] != aligned["label"]
    )
    denominator = int(clean_positive_correct.sum())
    numerator = int(flip_to_opposite.sum())
    return {
        "clean_correct_positive": denominator,
        "polarity_flip_count": numerator,
        "polarity_flip_rate": numerator / denominator if denominator else 0.0,
    }


def build_summary(metrics_df: pd.DataFrame, runs_root: Path, scenario_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    flip_rows = []

    for model_name, label in MODEL_LABELS.items():
        clean_row = metrics_df[(metrics_df["scenario"] == "clean") & (metrics_df["model"] == model_name)].iloc[0]
        clean_dir = runs_root / "clean" / model_name

        for scenario_name in scenario_names:
            noisy_row = metrics_df[(metrics_df["scenario"] == scenario_name) & (metrics_df["model"] == model_name)].iloc[0]
            if scenario_name == "clean":
                flip_info = {
                    "clean_correct_positive": 0,
                    "polarity_flip_count": 0,
                    "polarity_flip_rate": 0.0,
                }
            else:
                flip_info = compute_polarity_flip_rate(clean_dir, runs_root / scenario_name / model_name)

            flip_rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_label": noisy_row["scenario_label"],
                    "scenario_family": noisy_row["scenario_family"],
                    "noise_level": int(noisy_row["noise_level"]),
                    "model": model_name,
                    "model_label": label,
                    **flip_info,
                }
            )

            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_label": noisy_row["scenario_label"],
                    "scenario_family": noisy_row["scenario_family"],
                    "noise_level": int(noisy_row["noise_level"]),
                    "model": model_name,
                    "model_label": label,
                    "macro_f1_pct": float(noisy_row["macro_f1"]) * 100.0,
                    "aupr_pct": float(noisy_row["aupr_ovr"]) * 100.0,
                    "resistance_f1_pct": float(noisy_row["resistance_f1"]) * 100.0,
                    "delta_macro_f1_pct": (float(noisy_row["macro_f1"]) - float(clean_row["macro_f1"])) * 100.0,
                    "delta_aupr_pct": (float(noisy_row["aupr_ovr"]) - float(clean_row["aupr_ovr"])) * 100.0,
                    "polarity_flip_rate_pct": float(flip_info["polarity_flip_rate"]) * 100.0,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(flip_rows)


def plot_metric_curves(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plot_specs = [("macro_f1_pct", "Macro-F1 (%)"), ("polarity_flip_rate_pct", "Polarity flip rate (%)")]
    family_styles = {"pathway": "-", "combined": "--"}

    for ax, (column, ylabel) in zip(axes, plot_specs):
        for model_label in summary_df["model_label"].unique():
            model_df = summary_df[summary_df["model_label"] == model_label]
            clean_y = float(model_df.loc[model_df["scenario"] == "clean", column].iloc[0])
            for family in ["pathway", "combined"]:
                family_df = model_df[model_df["scenario_family"] == family].sort_values("noise_level")
                x = [0] + family_df["noise_level"].tolist()
                y = [clean_y] + family_df[column].tolist()
                ax.plot(x, y, marker="o", linestyle=family_styles[family], label=f"{model_label} ({family})")
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 10, 20])
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output_dir / "robustness_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "robustness_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scenario-order", type=str, default="clean,pathway10,pathway20,combined10,combined20")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_names = [item.strip() for item in args.scenario_order.split(",") if item.strip()]

    metrics_df = load_metrics_rows(runs_root, scenario_names)
    summary_df, flip_df = build_summary(metrics_df, runs_root, scenario_names)

    metrics_df.to_csv(output_dir / "raw_metrics_summary.csv", index=False)
    summary_df.to_csv(output_dir / "robustness_summary.csv", index=False)
    flip_df.to_csv(output_dir / "polarity_flip_summary.csv", index=False)
    plot_metric_curves(summary_df, output_dir)

    overview = {
        "runs_root": str(runs_root),
        "scenarios": scenario_names,
        "output_files": [
            "raw_metrics_summary.csv",
            "robustness_summary.csv",
            "polarity_flip_summary.csv",
            "robustness_curves.png",
            "robustness_curves.pdf",
        ],
    }
    (output_dir / "analysis_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
