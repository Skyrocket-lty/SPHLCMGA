from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


PROB_COLUMNS = ["prob_non_association", "prob_resistance", "prob_sensitivity"]
POSITIVE_LABELS = {1, 2}


def load_predictions_dir(result_dir: Path) -> dict[str, pd.DataFrame]:
    fold_dirs = sorted([path for path in result_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {result_dir}")

    fold_frames: dict[str, pd.DataFrame] = {}
    for fold_dir in fold_dirs:
        prediction_path = fold_dir / "validation_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing validation_predictions.csv under {fold_dir}")
        frame = pd.read_csv(prediction_path)
        frame["fold"] = fold_dir.name
        fold_frames[fold_dir.name] = frame
    return fold_frames


def validation_keys(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "sample_id",
        "ncrna_id",
        "drug_id",
        "disease_id",
        "label",
        "split_role",
        "source_dataset",
    ]
    present = [column for column in candidates if column in frame.columns]
    if len(present) < 4:
        raise ValueError("Could not infer a stable sample key from validation predictions.")
    return present


def shared_validation_keys(standard: pd.DataFrame, rich: pd.DataFrame) -> list[str]:
    standard_keys = validation_keys(standard)
    rich_keys = validation_keys(rich)
    if "sample_id" in standard_keys and "sample_id" in rich_keys:
        return ["sample_id"]
    shared = [column for column in standard_keys if column in rich_keys and column != "sample_id"]
    if len(shared) < 4:
        raise ValueError(f"Could not infer shared validation keys: {standard_keys} vs {rich_keys}")
    return shared


def ensure_unique_keys(frame: pd.DataFrame, key_cols: list[str], tag: str) -> None:
    dup_mask = frame.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        dup_rows = frame.loc[dup_mask, key_cols].head(5).to_dict(orient="records")
        raise ValueError(f"Duplicate validation keys detected in {tag}: {dup_rows}")


def attach_occurrence_index(frame: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["_occ_idx"] = enriched.groupby(key_cols).cumcount()
    return enriched


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["label"].to_numpy()
    preds = frame["pred_label"].to_numpy()
    probs = frame[PROB_COLUMNS].to_numpy()
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "resistance_f1": float(f1_score(y_true, preds, labels=[1], average="macro", zero_division=0)),
        "auc_ovr": float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")),
        "aupr_ovr": float(average_precision_score(y_bin, probs, average="macro")),
    }


def summarize_fold_metrics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fold_name, frame in sorted(frames.items()):
        metrics = compute_metrics(frame)
        metrics["fold"] = fold_name
        rows.append(metrics)
    return pd.DataFrame(rows)


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * cumulative))


def parse_expected_metrics(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    values = [float(item.strip()) for item in raw.split(",")]
    if len(values) != 4:
        raise ValueError("--expected-metrics expects four comma-separated values: accuracy,macro_f1,auc,aupr")
    return {
        "accuracy": values[0] / 100.0,
        "macro_f1": values[1] / 100.0,
        "auc_ovr": values[2] / 100.0,
        "aupr_ovr": values[3] / 100.0,
    }


def compare_to_expected(summary: pd.DataFrame, expected: dict[str, float] | None, tolerance: float) -> pd.DataFrame:
    rows = []
    mean_row = summary.mean(numeric_only=True).to_dict()
    for key in ["accuracy", "macro_f1", "auc_ovr", "aupr_ovr"]:
        actual = float(mean_row[key])
        expected_value = None if expected is None else float(expected[key])
        delta = None if expected_value is None else actual - expected_value
        within_tolerance = None if expected_value is None else abs(delta) <= tolerance
        rows.append(
            {
                "metric": key,
                "actual_mean": actual,
                "expected_mean": expected_value,
                "delta": delta,
                "within_tolerance": within_tolerance,
            }
        )
    return pd.DataFrame(rows)


def align_predictions(
    standard_frames: dict[str, pd.DataFrame],
    rich_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    aligned_parts: list[pd.DataFrame] = []
    folds = sorted(set(standard_frames) | set(rich_frames))
    if set(standard_frames) != set(rich_frames):
        raise ValueError(f"Fold mismatch between standard and rich results: {folds}")

    for fold_name in folds:
        standard = standard_frames[fold_name].copy()
        rich = rich_frames[fold_name].copy()
        key_cols = shared_validation_keys(standard, rich)
        standard_with_idx = attach_occurrence_index(standard, key_cols)
        rich_with_idx = attach_occurrence_index(rich, key_cols)

        merged = standard_with_idx.merge(
            rich_with_idx,
            on=key_cols + ["_occ_idx"],
            suffixes=("_standard", "_rich"),
            how="inner",
        )
        if len(merged) != len(standard) or len(merged) != len(rich):
            raise ValueError(f"Validation samples do not align one-to-one in {fold_name}")
        merged["fold"] = fold_name
        aligned_parts.append(merged)

    aligned = pd.concat(aligned_parts, ignore_index=True)
    for base_column in ["label", "ncrna_id", "drug_id", "disease_id", "split_role", "source_dataset"]:
        if base_column not in aligned.columns:
            standard_col = f"{base_column}_standard"
            rich_col = f"{base_column}_rich"
            if standard_col in aligned.columns:
                aligned[base_column] = aligned[standard_col]
                if rich_col in aligned.columns and not aligned[standard_col].equals(aligned[rich_col]):
                    raise ValueError(f"Mismatched {base_column} values after aligning predictions.")
            elif rich_col in aligned.columns:
                aligned[base_column] = aligned[rich_col]
    aligned["standard_correct"] = aligned["pred_label_standard"] == aligned["label"]
    aligned["rich_correct"] = aligned["pred_label_rich"] == aligned["label"]

    transitions = []
    for standard_correct, rich_correct in zip(aligned["standard_correct"], aligned["rich_correct"]):
        if not standard_correct and rich_correct:
            transitions.append("wrong_to_right")
        elif standard_correct and not rich_correct:
            transitions.append("right_to_wrong")
        elif standard_correct and rich_correct:
            transitions.append("right_to_right")
        else:
            transitions.append("wrong_to_wrong")
    aligned["transition"] = transitions

    standard_prob_matrix = aligned[[f"{column}_standard" for column in PROB_COLUMNS]].to_numpy()
    sorted_probs = -(-standard_prob_matrix)  # copy
    sorted_probs.sort(axis=1)
    aligned["standard_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    return aligned


def build_transition_summary(aligned: pd.DataFrame) -> pd.DataFrame:
    total = len(aligned)
    transition_counts = aligned["transition"].value_counts().reindex(
        ["wrong_to_right", "right_to_wrong", "wrong_to_wrong", "right_to_right"],
        fill_value=0,
    )
    standard_wrong = int((~aligned["standard_correct"]).sum())
    standard_right = int(aligned["standard_correct"].sum())
    correction_count = int(transition_counts["wrong_to_right"])
    regression_count = int(transition_counts["right_to_wrong"])
    summary = {
        "total_samples": total,
        "standard_wrong": standard_wrong,
        "standard_right": standard_right,
        "rich_corrections": correction_count,
        "rich_regressions": regression_count,
        "correction_rate": correction_count / standard_wrong if standard_wrong else 0.0,
        "regression_rate": regression_count / standard_right if standard_right else 0.0,
        "net_correction": correction_count - regression_count,
        "mcnemar_pvalue": exact_mcnemar_pvalue(correction_count, regression_count),
    }
    rows = []
    for transition_name, count in transition_counts.items():
        rows.append({"transition": transition_name, "count": int(count), "fraction": int(count) / total if total else 0.0})
    transition_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame([summary])
    return transition_df, summary_df


def build_hard_case_summary(aligned: pd.DataFrame, switch_pairs_df: pd.DataFrame | None) -> pd.DataFrame:
    standard_wrong = ~aligned["standard_correct"]
    polarity_confusion = (
        aligned["label"].isin(POSITIVE_LABELS)
        & aligned["pred_label_standard"].isin(POSITIVE_LABELS)
        & (aligned["pred_label_standard"] != aligned["label"])
    )
    positive_class_errors = aligned["label"].isin(POSITIVE_LABELS) & standard_wrong

    wrong_margin = aligned.loc[standard_wrong, "standard_margin"]
    margin_threshold = float(wrong_margin.quantile(0.25)) if len(wrong_margin) else 0.0
    low_margin_errors = standard_wrong & (aligned["standard_margin"] <= margin_threshold)

    subsets: list[tuple[str, pd.Series]] = [
        ("polarity_confusion", polarity_confusion),
        ("positive_class_errors", positive_class_errors),
        ("low_margin_errors", low_margin_errors),
    ]

    if switch_pairs_df is not None:
        switch_cols = [column for column in ["ncrna_id", "drug_id", "disease_id"] if column in switch_pairs_df.columns]
        if len(switch_cols) >= 2:
            switch_frame = aligned.merge(
                switch_pairs_df[switch_cols].drop_duplicates(),
                on=switch_cols,
                how="left",
                indicator=True,
            )
            subsets.append(("switch_pair_related", switch_frame["_merge"].eq("both")))

    rows = []
    for subset_name, mask in subsets:
        subset = aligned.loc[mask].copy()
        total = len(subset)
        if total == 0:
            rows.append(
                {
                    "subset": subset_name,
                    "n_samples": 0,
                    "standard_accuracy": None,
                    "rich_accuracy": None,
                    "correction_rate": None,
                    "regression_rate": None,
                    "net_correction": None,
                    "mcnemar_pvalue": None,
                    "margin_threshold": margin_threshold if subset_name == "low_margin_errors" else None,
                }
            )
            continue

        corrections = int((subset["transition"] == "wrong_to_right").sum())
        regressions = int((subset["transition"] == "right_to_wrong").sum())
        standard_wrong_subset = int((~subset["standard_correct"]).sum())
        standard_right_subset = int(subset["standard_correct"].sum())
        rows.append(
            {
                "subset": subset_name,
                "n_samples": total,
                "standard_accuracy": float(subset["standard_correct"].mean()),
                "rich_accuracy": float(subset["rich_correct"].mean()),
                "correction_rate": corrections / standard_wrong_subset if standard_wrong_subset else 0.0,
                "regression_rate": regressions / standard_right_subset if standard_right_subset else 0.0,
                "net_correction": corrections - regressions,
                "mcnemar_pvalue": exact_mcnemar_pvalue(corrections, regressions),
                "margin_threshold": margin_threshold if subset_name == "low_margin_errors" else None,
            }
        )
    return pd.DataFrame(rows)


def save_transition_plot(transition_df: pd.DataFrame, output_path: Path) -> None:
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator

    order = ["wrong_to_right", "right_to_wrong", "wrong_to_wrong", "right_to_right"]
    labels = ["Wrong->Right", "Right->Wrong", "Wrong->Wrong", "Right->Right"]
    color_map = {
        "wrong_to_right": "#E57373",
        "right_to_wrong": "#B0BEC5",
        "wrong_to_wrong": "#90A4AE",
        "right_to_right": "#5C6BC0",
    }
    edge_map = {
        "wrong_to_right": "#D35F5F",
        "right_to_wrong": "#9EABB1",
        "wrong_to_wrong": "#7D929D",
        "right_to_right": "#4E5CB2",
    }
    counts = [int(transition_df.loc[transition_df["transition"] == key, "count"].iloc[0]) for key in order]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.9, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    container = FancyBboxPatch(
        (0.00, 0.00),
        1.00,
        1.00,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor="white",
        edgecolor="#D8DEE8",
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(container)

    y = list(range(len(labels)))
    heights = [0.62, 0.56, 0.56, 0.56]
    bars = []
    for idx, (yy, count, key, height) in enumerate(zip(y, counts, order, heights)):
        linewidth = 1.25 if idx == 0 else 0.95
        bar = ax.barh(
            yy,
            count,
            color=color_map[key],
            edgecolor=edge_map[key],
            linewidth=linewidth,
            height=height,
            zorder=3,
        )[0]
        bars.append(bar)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.6, color="#334155")
    for tick in ax.get_yticklabels():
        tick.set_horizontalalignment("right")
    ax.tick_params(axis="y", length=0, pad=10)
    ax.invert_yaxis()

    ax.set_xlabel("Sample count", fontsize=9.2, color="#475569")
    ax.set_title("Prediction transition counts", color="#334155", fontsize=10.8, pad=6, weight="semibold")

    xmax = max(counts) * 1.22
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.tick_params(axis="x", labelsize=8.5, colors="#475569", width=0.6, length=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["bottom"].set_linewidth(0.7)

    ax.grid(axis="y", color="#E5E7EB", alpha=0.50, linewidth=0.6)
    ax.set_axisbelow(True)

    for idx, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_width() + xmax * 0.020,
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            va="center",
            ha="left",
            fontsize=9.2,
            fontweight="bold" if idx == 0 else "semibold",
            color="#334155",
            zorder=4,
        )

    fig.subplots_adjust(left=0.28, right=0.96, top=0.89, bottom=0.14)
    fig.savefig(output_path, dpi=600, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def save_hard_case_plot(hard_case_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = hard_case_df[hard_case_df["n_samples"] > 0].copy()
    if plot_df.empty:
        return

    labels = {
        "polarity_confusion": "Polarity confusion",
        "positive_class_errors": "Positive-class errors",
        "low_margin_errors": "Low-margin errors",
        "switch_pair_related": "Switch-pair related",
    }
    color_map = {
        "polarity_confusion": "#8DA0CB",      # slate-blue
        "positive_class_errors": "#4DBBD5",   # teal
        "low_margin_errors": "#E64B35",       # coral
        "switch_pair_related": "#B3B3B3",     # neutral gray
    }
    edge_map = {
        "polarity_confusion": "#7186B3",
        "positive_class_errors": "#319AB4",
        "low_margin_errors": "#C33E29",
        "switch_pair_related": "#969696",
    }

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xlabels = [labels.get(name, name) for name in plot_df["subset"]]
    keys = plot_df["subset"].tolist()
    values = plot_df["correction_rate"].astype(float).tolist()
    bars = ax.bar(
        xlabels,
        values,
        color=[color_map.get(k, "#8DA0CB") for k in keys],
        edgecolor=[edge_map.get(k, "#7186B3") for k in keys],
        linewidth=0.9,
        width=0.56,
        zorder=3,
    )

    ax.set_ylabel("Correction rate", fontsize=10.0, color="#334155")
    ax.set_ylim(0.0, max(0.05, float(plot_df["correction_rate"].max()) * 1.18))
    ax.set_title("Hard-case correction rate", fontsize=11.0, color="#334155", pad=6, weight="semibold")
    ax.tick_params(axis="x", labelsize=9.0, colors="#475569", rotation=0)
    ax.tick_params(axis="y", labelsize=8.7, colors="#475569")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.grid(axis="y", color="#E5E7EB", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ax.get_ylim()[1] * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="semibold",
            color="#334155",
        )

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=600, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Differential error-correction analysis for SPHLCMGA variants.")
    parser.add_argument("--standard-dir", required=True, type=str)
    parser.add_argument("--rich-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--expected-standard-metrics", type=str, default=None)
    parser.add_argument("--expected-rich-metrics", type=str, default=None)
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument("--switch-pairs-csv", type=str, default=None)
    args = parser.parse_args()

    standard_dir = Path(args.standard_dir)
    rich_dir = Path(args.rich_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_frames = load_predictions_dir(standard_dir)
    rich_frames = load_predictions_dir(rich_dir)
    standard_summary = summarize_fold_metrics(standard_frames)
    rich_summary = summarize_fold_metrics(rich_frames)

    standard_summary.to_csv(output_dir / "standard_fold_metrics.csv", index=False)
    rich_summary.to_csv(output_dir / "rich_fold_metrics.csv", index=False)
    compare_to_expected(standard_summary, parse_expected_metrics(args.expected_standard_metrics), args.tolerance).to_csv(
        output_dir / "standard_metric_check.csv",
        index=False,
    )
    compare_to_expected(rich_summary, parse_expected_metrics(args.expected_rich_metrics), args.tolerance).to_csv(
        output_dir / "rich_metric_check.csv",
        index=False,
    )

    aligned = align_predictions(standard_frames, rich_frames)
    aligned.to_csv(output_dir / "aligned_oof_predictions.csv", index=False)

    transition_df, summary_df = build_transition_summary(aligned)
    transition_df.to_csv(output_dir / "transition_counts.csv", index=False)
    summary_df.to_csv(output_dir / "differential_summary.csv", index=False)

    switch_pairs_df = None
    if args.switch_pairs_csv:
        switch_pairs_df = pd.read_csv(args.switch_pairs_csv)
    hard_case_df = build_hard_case_summary(aligned, switch_pairs_df)
    hard_case_df.to_csv(output_dir / "hard_case_summary.csv", index=False)

    save_transition_plot(transition_df, output_dir / "transition_counts.png")
    save_hard_case_plot(hard_case_df, output_dir / "hard_case_correction_rate.png")

    overview = {
        "standard_dir": str(standard_dir),
        "rich_dir": str(rich_dir),
        "n_samples": int(len(aligned)),
        "standard_mean_metrics": standard_summary.mean(numeric_only=True).to_dict(),
        "rich_mean_metrics": rich_summary.mean(numeric_only=True).to_dict(),
    }
    (output_dir / "analysis_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
