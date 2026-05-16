import gzip
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyucell
import requests
import scipy.io
import scipy.sparse as sp
from scipy.stats import mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parent
OUTDIR = REPO_ROOT / "results" / "polarity_switch"
CACHE = OUTDIR / "scrna_cache"

LUNG_RAW_TAR = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE127nnn/GSE127465/suppl/GSE127465_RAW.tar"
LUNG_META = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE127nnn/GSE127465/suppl/GSE127465_human_cell_metadata_54773x25.tsv.gz"

PDAC_FILELIST = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154778/suppl/filelist.txt"
PDAC_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154778/suppl"
PDAC_RAW_TAR = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154778/suppl/GSE154778_RAW.tar"

MIR21_TARGETS = [
    "PTEN",
    "PDCD4",
    "RECK",
    "SPRY2",
    "BTG2",
    "TGFBR2",
    "RHOB",
    "TPM1",
    "FASLG",
    "APAF1",
    "TIMP3",
]
ABC_MODULE = ["ABCB1", "ABCC1", "ABCC2", "ABCC3", "ABCG2"]
SUPPORT_MODULE = ["STAT3", "PTEN", "AKT1", "BCL2", "CASP3", "SLC7A11", "TFRC"]

CELLTYPE_MARKERS = {
    "malignant": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT17", "MSLN", "CEACAM6"],
    "macrophage_monocyte": ["LST1", "TYROBP", "C1QA", "C1QB", "FCER1G", "CTSS"],
    "t_cell": ["CD3D", "CD3E", "TRBC1", "TRBC2", "IL7R", "LTB"],
    "fibroblast_caf": ["COL1A1", "COL1A2", "LUM", "DCN", "COL3A1", "TAGLN"],
    "endothelial": ["PECAM1", "VWF", "KDR", "EMCN", "ENG", "RAMP2"],
}

MODULE_SIGNATURES = {
    "miR21_target": MIR21_TARGETS,
    "abc_transporter": ABC_MODULE,
    "cisplatin_support": SUPPORT_MODULE,
}


def bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = ranked[idx] * n / rank
        prev = min(prev, value)
        adjusted[idx] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def ensure_download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with path.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return path


def map_lung_celltype(major):
    major = str(major)
    if major in {"Type I cells", "Type II cells", "Club cells", "Ciliated cells"} or major.startswith("Patient"):
        return "malignant"
    if major == "tMoMacDC":
        return "macrophage_monocyte"
    if major == "tT cells":
        return "t_cell"
    if major == "Fibroblasts":
        return "fibroblast_caf"
    if major == "Endothelial cells":
        return "endothelial"
    return None


def compute_sample_ucell(adata, signatures, chunk_size=256):
    pyucell.compute_ucell_scores(adata, signatures=signatures, chunk_size=chunk_size, n_jobs=1)
    return adata


def summarize_cells(obs_df, context_name):
    summaries = []
    proportions = []
    valid = obs_df[obs_df["broad_cell_type"].notna()].copy()
    for (patient_id, cell_type), sub in valid.groupby(["patient_id", "broad_cell_type"]):
        summaries.append(
            {
                "context": context_name,
                "patient_id": patient_id,
                "cell_type": cell_type,
                "cell_count": int(len(sub)),
                "miR21_target_UCell": float(sub["miR21_target_UCell"].median()),
                "abc_transporter_UCell": float(sub["abc_transporter_UCell"].median()),
                "cisplatin_support_UCell": float(sub["cisplatin_support_UCell"].median()),
            }
        )
    for patient_id, sub in valid.groupby("patient_id"):
        total = len(sub)
        for cell_type, count in sub["broad_cell_type"].value_counts().items():
            proportions.append(
                {
                    "context": context_name,
                    "patient_id": patient_id,
                    "cell_type": cell_type,
                    "cell_count": int(count),
                    "cell_fraction": float(count / total),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(proportions)


def load_lung_context():
    raw_tar = ensure_download(LUNG_RAW_TAR, CACHE / "GSE127465_RAW.tar")
    meta_path = ensure_download(LUNG_META, CACHE / "GSE127465_human_cell_metadata.tsv.gz")
    meta = pd.read_csv(meta_path, sep="\t")
    meta = meta[meta["Tissue"] == "tumor"].copy()
    meta["broad_cell_type"] = meta["Major cell type"].map(map_lung_celltype)
    meta = meta[meta["broad_cell_type"].notna()].copy()

    cell_frames = []
    with tarfile.open(raw_tar, "r") as tf:
        members = [m for m in tf.getmembers() if "human_" in m.name and "_raw_counts.tsv.gz" in m.name and "t" in m.name]
        for member in members:
            library = member.name.split("human_")[1].split("_raw_counts")[0]
            sub_meta = meta[meta["Library"] == library].copy()
            if sub_meta.empty:
                continue
            selected = set(sub_meta["Barcode"].astype(str))
            handle = tf.extractfile(member)
            if handle is None:
                continue
            with gzip.GzipFile(fileobj=handle) as gz:
                chunk_iter = pd.read_csv(gz, sep="\t", chunksize=256)
                for chunk in chunk_iter:
                    hit = chunk[chunk["barcode"].astype(str).isin(selected)].copy()
                    if hit.empty:
                        continue
                    gene_cols = [col for col in hit.columns if col != "barcode"]
                    obs = sub_meta.set_index("Barcode").loc[hit["barcode"].astype(str)].copy()
                    obs = obs.rename(columns={"Patient": "patient_id"})
                    obs.index = [f"{library}_{bc}" for bc in hit["barcode"].astype(str)]
                    X = sp.csr_matrix(hit[gene_cols].to_numpy(dtype=np.float32))
                    adata = ad.AnnData(X=X, obs=obs[["patient_id", "Library", "broad_cell_type"]].copy(), var=pd.DataFrame(index=gene_cols))
                    adata = compute_sample_ucell(adata, MODULE_SIGNATURES)
                    cell_frames.append(adata.obs.reset_index(names="cell_id"))
    cell_df = pd.concat(cell_frames, ignore_index=True)
    summary_df, prop_df = summarize_cells(cell_df, "lung_resistance_context")
    return cell_df, summary_df, prop_df


def load_pdac_files():
    txt = requests.get(PDAC_FILELIST, timeout=60).text.splitlines()
    samples = defaultdict(dict)
    for line in txt:
        if not line.startswith("File\tGSM"):
            continue
        parts = line.split("\t")
        name = parts[1]
        if name.endswith("_barcodes.tsv.gz"):
            sample_id = name.replace("_barcodes.tsv.gz", "")
            samples[sample_id]["barcodes"] = f"{PDAC_BASE}/{name}"
        elif name.endswith("_genes.tsv.gz") or name.endswith("_features.tsv.gz"):
            sample_id = name.replace("_genes.tsv.gz", "").replace("_features.tsv.gz", "")
            samples[sample_id]["genes"] = f"{PDAC_BASE}/{name}"
        elif name.endswith("_matrix.mtx.gz"):
            sample_id = name.replace("_matrix.mtx.gz", "")
            samples[sample_id]["matrix"] = f"{PDAC_BASE}/{name}"
    return samples


def classify_pancreatic_cells(adata):
    pyucell.compute_ucell_scores(adata, signatures=CELLTYPE_MARKERS, chunk_size=512, n_jobs=1)
    score_cols = [f"{name}_UCell" for name in CELLTYPE_MARKERS]
    scores = adata.obs[score_cols].to_numpy()
    best = scores.argmax(axis=1)
    second = np.partition(scores, -2, axis=1)[:, -2]
    best_scores = scores[np.arange(scores.shape[0]), best]
    labels = []
    names = list(CELLTYPE_MARKERS.keys())
    for idx, score in enumerate(best_scores):
        margin = score - second[idx]
        if score < 0.10 or margin < 0.01:
            labels.append(None)
        else:
            labels.append(names[best[idx]])
    adata.obs["broad_cell_type"] = labels
    return adata


def load_pancreatic_context():
    raw_tar = ensure_download(PDAC_RAW_TAR, CACHE / "GSE154778_RAW.tar")
    cell_frames = []
    with tarfile.open(raw_tar, "r") as tf:
        sample_members = defaultdict(dict)
        for member in tf.getmembers():
            name = Path(member.name).name
            if name.endswith("_barcodes.tsv.gz"):
                sample_members[name.replace("_barcodes.tsv.gz", "")]["barcodes"] = member
            elif name.endswith("_genes.tsv.gz") or name.endswith("_features.tsv.gz"):
                sample_members[name.replace("_genes.tsv.gz", "").replace("_features.tsv.gz", "")]["genes"] = member
            elif name.endswith("_matrix.mtx.gz"):
                sample_members[name.replace("_matrix.mtx.gz", "")]["matrix"] = member

        for sample_key, members in sample_members.items():
            if not {"barcodes", "genes", "matrix"}.issubset(members):
                continue
            sample_alias = sample_key.split("_", 1)[1] if "_" in sample_key else sample_key

            with gzip.GzipFile(fileobj=tf.extractfile(members["barcodes"])) as handle:
                barcodes = pd.read_csv(handle, header=None)[0].astype(str).tolist()
            with gzip.GzipFile(fileobj=tf.extractfile(members["genes"])) as handle:
                genes_df = pd.read_csv(handle, header=None, sep="\t")
            genes = genes_df.iloc[:, 1 if genes_df.shape[1] > 1 else 0].astype(str).tolist()
            with gzip.GzipFile(fileobj=tf.extractfile(members["matrix"])) as handle:
                matrix = scipy.io.mmread(handle).tocsr().transpose().astype(np.float32)

            obs = pd.DataFrame(index=[f"{sample_alias}:{bc}" for bc in barcodes])
            obs["patient_id"] = sample_alias
            adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
            adata.var_names_make_unique()
            classify_pancreatic_cells(adata)
            compute_sample_ucell(adata, MODULE_SIGNATURES, chunk_size=512)
            cell_frames.append(adata.obs.reset_index(names="cell_id"))

    cell_df = pd.concat(cell_frames, ignore_index=True)
    summary_df, prop_df = summarize_cells(cell_df, "pancreatic_sensitivity_context")
    return cell_df, summary_df, prop_df


def compare_contexts(summary_df):
    rows = []
    for metric in ["miR21_target_UCell", "abc_transporter_UCell", "cisplatin_support_UCell"]:
        for cell_type in ["malignant", "macrophage_monocyte", "fibroblast_caf", "t_cell", "endothelial"]:
            sub = summary_df[summary_df["cell_type"] == cell_type]
            x = sub[sub["context"] == "lung_resistance_context"][metric].dropna().to_numpy()
            y = sub[sub["context"] == "pancreatic_sensitivity_context"][metric].dropna().to_numpy()
            if len(x) < 3 or len(y) < 3:
                p_value = np.nan
                stat = np.nan
            else:
                stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "metric": metric,
                    "cell_type": cell_type,
                    "lung_n": int(len(x)),
                    "pancreatic_n": int(len(y)),
                    "lung_median": float(np.median(x)) if len(x) else np.nan,
                    "pancreatic_median": float(np.median(y)) if len(y) else np.nan,
                    "median_difference": float(np.median(x) - np.median(y)) if len(x) and len(y) else np.nan,
                    "mannwhitney_u": float(stat) if not np.isnan(stat) else np.nan,
                    "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    result.loc[valid, "fdr_p"] = bh_fdr(result.loc[valid, "p_value"].tolist())
    return result


def compare_composition(prop_df):
    rows = []
    for cell_type in ["malignant", "macrophage_monocyte", "fibroblast_caf", "t_cell", "endothelial"]:
        sub = prop_df[prop_df["cell_type"] == cell_type]
        x = sub[sub["context"] == "lung_resistance_context"]["cell_fraction"].dropna().to_numpy()
        y = sub[sub["context"] == "pancreatic_sensitivity_context"]["cell_fraction"].dropna().to_numpy()
        if len(x) < 3 or len(y) < 3:
            stat = np.nan
            p_value = np.nan
        else:
            stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
        rows.append(
            {
                "cell_type": cell_type,
                "lung_n": int(len(x)),
                "pancreatic_n": int(len(y)),
                "lung_mean_fraction": float(np.mean(x)) if len(x) else np.nan,
                "pancreatic_mean_fraction": float(np.mean(y)) if len(y) else np.nan,
                "mean_difference": float(np.mean(x) - np.mean(y)) if len(x) and len(y) else np.nan,
                "mannwhitney_u": float(stat) if not np.isnan(stat) else np.nan,
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    result.loc[valid, "fdr_p"] = bh_fdr(result.loc[valid, "p_value"].tolist())
    return result


def write_gene_manifest(lung_cell_df, pdac_cell_df):
    used = []
    for context, cell_df in [("lung_resistance_context", lung_cell_df), ("pancreatic_sensitivity_context", pdac_cell_df)]:
        for module_name, genes in MODULE_SIGNATURES.items():
            present = [gene for gene in genes if f"{module_name}_UCell" in cell_df.columns]
            used.append(
                {
                    "context": context,
                    "module": module_name,
                    "genes_requested": " | ".join(genes),
                    "genes_used_note": "UCell scored against the genes available in the corresponding GEO matrix.",
                }
            )
    pd.DataFrame(used).to_csv(OUTDIR / "miR21_target_genes_used.csv", index=False)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    lung_cells, lung_summary, lung_props = load_lung_context()
    pdac_cells, pdac_summary, pdac_props = load_pancreatic_context()

    summary_df = pd.concat([lung_summary, pdac_summary], ignore_index=True)
    prop_df = pd.concat([lung_props, pdac_props], ignore_index=True)
    stats_df = compare_contexts(summary_df)
    composition_df = compare_composition(prop_df)

    summary_df.to_csv(OUTDIR / "scrna_module_scores_by_patient.csv", index=False)
    prop_df.to_csv(OUTDIR / "scrna_celltype_proportions.csv", index=False)
    stats_df.to_csv(OUTDIR / "scrna_module_stats.csv", index=False)
    composition_df.to_csv(OUTDIR / "scrna_composition_stats.csv", index=False)
    write_gene_manifest(lung_cells, pdac_cells)

    summary = {
        "lung_patient_count": int(lung_summary["patient_id"].nunique()),
        "pancreatic_patient_count": int(pdac_summary["patient_id"].nunique()),
        "lung_celltypes": sorted([x for x in lung_summary["cell_type"].dropna().unique().tolist()]),
        "pancreatic_celltypes": sorted([x for x in pdac_summary["cell_type"].dropna().unique().tolist()]),
        "contexts": ["lung_resistance_context", "pancreatic_sensitivity_context"],
        "datasets": {
            "lung": "GSE127465",
            "pancreatic": "GSE154778",
        },
    }
    (OUTDIR / "scrna_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


