import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

TASKS = {
    "dataset1_cv": ["dataset1_maintest.py"],
    "dataset2_cv": ["dataset2_maintest.py"],
    "cold_start": ["cold_start_runner.py"],
    "interpretability": ["interpretability_runner.py"],
    "knockout": ["dataset2_knockout_analysis.py"],
    "polarity_switch": ["polarity_switch_hcmg_analysis.py"],
    "omics_support": ["omics_support_analysis.py"],
    "patient_survival": ["patient_survival_analysis.py"],
    "scrna_proxy": ["polarity_switch_scrna_analysis.py"],
    "external_validation_prepare": ["external_validation/prepare_external_validation.py"],
    "external_validation_run": ["external_validation/run_external_validation.py"],
    "large_context_study": ["large_context/run_sphlcMga_pathway_context_study.py"],
    "large_context_evidence_trace": ["large_context/trace_pathway_context_evidence.py"],
    "large_context_ko_consistency": ["large_context/run_pathway_context_ko_consistency.py"],
    "large_context_prior_noise": ["large_context/run_sphlcMga_prior_noise_robustness.py"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Task dispatcher for the public SPHLCMGA repository")
    parser.add_argument("--task", choices=sorted(TASKS.keys()), required=True, help="Task name to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Extra arguments passed to the selected script")
    return parser.parse_args()


def main():
    args = parse_args()
    script = REPO_ROOT / TASKS[args.task][0]
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    cmd = [sys.executable, str(script)] + forwarded
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
