#!/usr/bin/env python3
"""
Log-reg pipeline to predict **Fusion status** (T vs F) from WSI tile features
using 5-fold stratified *patient-grouped* cross-validation.

Main points
-----------
1.  A single CSV (“slides_with_fusion_status.csv”) maps SlideID → Fusion status.
2.  `attach_fusion_status()` injects that label into each metadata table,
    encodes F→0 , T→1, and drops NA rows.
3.  All evaluation relies on `run_logistic_regression`, which performs
    StratifiedGroupKFold CV (splits=5 by default).
4.  No Contributor-specific splits remain.

Author: <you>
"""

# ───────────────────── imports ────────────────────────────────────────────
import os
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.colors as mcolors

# Illustrator-friendly font setup
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

# ───────────────────── globals ────────────────────────────────────────────
LABEL_COL   = "Fusion status"
LOOKUP_CSV  = "slides_with_fusion_status.csv"        # truth table
BASE_PATH   = "/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/sarcoma-features/"

# 0) build lookup (SlideID stripped of any '.oidX' suffix) ─────────────────
FUSION_LOOKUP = (
    pd.read_csv(LOOKUP_CSV)
      .assign(SlideID=lambda d: d["SlideID"].astype(str).str.split(".").str[0])
      .set_index("SlideID")[LABEL_COL]
      .astype(str)
)

# ───────────────────── helpers ────────────────────────────────────────────
def attach_fusion_status(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds numeric Fusion-status (F→0, T→1); drops rows w/ NA; returns copy.
    Assumes Tissue ID column contains '…oidX' suffix.
    """
    slide_base = meta_df["Tissue ID"].astype(str).str.split(".").str[0]
    meta_df[LABEL_COL] = slide_base.map(FUSION_LOOKUP).fillna("NA")
    meta_df = meta_df[meta_df[LABEL_COL].isin(["T", "F"])].copy()
    meta_df[LABEL_COL] = meta_df[LABEL_COL].map({"F": 0, "T": 1}).astype(int)
    return meta_df


# ───────────────────── 1) read .h5ad ──────────────────────────────────────
def process_h5ad(file_path):
    """Return (metadata_df, feature_df) from an .h5ad file."""
    with h5py.File(file_path, "r") as f:
        # ----- obs -----
        metadata = {
            k: f[f'obs/{k}'][:]
            for k in f['obs'].keys()
            if isinstance(f[f'obs/{k}'], h5py.Dataset)
        }
        meta_df = pd.DataFrame(metadata)

        # categorical → strings
        for cat_key in ("Contributor", "Patient ID"):
            if cat_key in f['obs']:
                try:
                    cats  = [x.decode() if isinstance(x, bytes) else x
                             for x in f[f'obs/{cat_key}/categories'][:]]
                    codes = f[f'obs/{cat_key}/codes'][:]
                    meta_df[cat_key] = [cats[c] for c in codes]
                except Exception as e:
                    print(f"[Warn] decode {cat_key}: {e}")

        # ----- X -----
        X = f['X'][:]
        var_names = f['var/_index'][:].astype(str)

    feat_df = pd.DataFrame(X, columns=var_names)

    # tidy Tissue ID
    if 'Tissue ID' in meta_df.columns:
        meta_df['Tissue ID'] = (
            meta_df['Tissue ID'].astype(str)
                                .str.replace(r"^b'|'$", "", regex=True)
        )
        meta_df.index = meta_df['Tissue ID']
        feat_df.index = meta_df.index

    return meta_df, feat_df


# ───────────────────── 2) single file wrapper ─────────────────────────────
def process_single_file(file_path):
    print(f"\n[process_single_file] {file_path}")
    meta, feat = process_h5ad(file_path)
    if meta.empty or feat.empty:
        return pd.DataFrame()

    df = pd.concat([meta, feat], axis=1)
    df = attach_fusion_status(df)
    df = df[df['Tissue ID'].str.endswith('.oid0')]          # canonical tiles
    print(f"   kept {df.shape[0]} rows")
    return df


# ───────────────────── 3) combine two feature sets ────────────────────────
def process_combination(file1, file2):
    print(f"\n[process_combination]\n  {file1}\n  {file2}")
    meta1, feat1 = process_h5ad(file1)
    meta2, feat2 = process_h5ad(file2)

    idx = meta1.index.intersection(meta2.index)
    if len(idx) == 0:
        return pd.DataFrame()

    meta  = meta1.loc[idx].copy()
    feats = pd.concat([feat1.loc[idx], feat2.loc[idx]], axis=1)
    df    = pd.concat([meta, feats], axis=1)

    df = attach_fusion_status(df)
    df = df[df['Tissue ID'].str.endswith('.oid0')]
    print(f"   combo rows: {df.shape[0]}")
    return df


# ───────────────────── 4) core CV routine ────────────────────────────────
def run_logistic_regression(df, splits=5, num_iterations=1):
    feat_cols = df.select_dtypes(include=[np.number]).columns.drop(LABEL_COL)
    X = df[feat_cols].to_numpy()
    y = df[LABEL_COL].to_numpy().astype(int)
    groups = df['Patient ID'].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=None)
    aucs, cms, metrics_all = [], [], []

    for _ in range(num_iterations):
        for tr_idx, te_idx in sgkf.split(X, y, groups):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # guard single-class folds
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            # t-test feature selection
            _, p = sp.stats.ttest_ind(
                X_tr[y_tr == 0], X_tr[y_tr == 1], axis=0, equal_var=False
            )
            sel = np.where(p < 0.05)[0]
            if sel.size == 0:
                sel = np.arange(X_tr.shape[1])

            model = LR(penalty='l1', C=100, class_weight='balanced',
                       solver='liblinear', max_iter=2000)
            model.fit(X_tr[:, sel], y_tr)

            probs = model.predict_proba(X_te[:, sel])[:, 1]
            preds = model.predict(X_te[:, sel])

            cms.append(confusion_matrix(y_te, preds))
            metrics_all.append({
                'precision': precision_score(y_te, preds, zero_division=0),
                'recall'   : recall_score(y_te, preds, zero_division=0),
                'f1'       : f1_score(y_te, preds, zero_division=0),
                'accuracy' : accuracy_score(y_te, preds)
            })
            fpr, tpr, _ = roc_curve(y_te, probs)
            aucs.append(auc(fpr, tpr))

    if not aucs:
        return None, None, None, {}, []

    return (np.mean(aucs), np.std(aucs, ddof=1),
            np.sum(cms, axis=0), pd.DataFrame(metrics_all).mean().to_dict(),
            aucs)


# ───────────────────── 5) single-file experiments ────────────────────────
def test_single_files(backbones, file_ids):
    n_files, n_bbs = len(file_ids), len(backbones)
    means = np.full((n_files, n_bbs), np.nan)
    stds  = np.full((n_files, n_bbs), np.nan)
    dist  = {}

    for j, bb in enumerate(backbones):
        for i, fid in enumerate(file_ids):
            path = f"{BASE_PATH}ad_wsi.{bb}-{fid}.h5ad"
            df   = process_single_file(path)
            if df.empty:
                dist[(bb, fid)] = []
                continue
            m, s, _, _, aucs = run_logistic_regression(df, splits=5, num_iterations=1)
            dist[(bb, fid)], means[i, j], stds[i, j] = aucs, m, s
    return means, stds, dist


# ───────────────────── 6) combination experiments ────────────────────────
def test_combinations(backbone, file_ids):
    files = [f"{BASE_PATH}ad_wsi.{backbone}-{fid}.h5ad" for fid in file_ids]
    n = len(files)
    auc_mat = np.full((n, n), -1.0)
    std_mat = np.full((n, n), -1.0)
    cm_list, metrics_list = [], []

    for i, j in combinations(range(n), 2):
        df = process_combination(files[i], files[j])
        if df.empty:
            continue
        m, s, cm, metrics, _ = run_logistic_regression(df, splits=5, num_iterations=1)
        auc_mat[i, j] = auc_mat[j, i] = m
        std_mat[i, j] = std_mat[j, i] = s
        cm_list.append((f"{backbone}-{file_ids[i]}", f"{backbone}-{file_ids[j]}", cm))
        metrics_list.append((f"{backbone}-{file_ids[i]}", f"{backbone}-{file_ids[j]}", metrics))
    return auc_mat, std_mat, cm_list, metrics_list


# ───────────────────── 7) plotting helpers (unchanged) ────────────────────
#   ... include your existing heat-map / box-plot / confusion-plot helpers ...
#########################
# 7) find_global_auc_min
#########################
def find_global_auc_min(single_aucs_matrix, combo_auc_dict):
    """
    Finds the minimum AUC across the single-file matrix
    AND all combination matrices for all backbones.
    combo_auc_dict is like {backbone: (auc_matrix, std_matrix)}.
    """
    all_aucs = []

    # Single-file
    all_aucs.extend(single_aucs_matrix.flatten())

    # Combo: replace -1 with NaN, gather valid entries
    for backbone, (auc_mat, _) in combo_auc_dict.items():
        auc_mat_valid = np.where(auc_mat == -1, np.nan, auc_mat)
        all_aucs.extend(auc_mat_valid.flatten())

    # Filter out NaNs
    all_aucs = [v for v in all_aucs if not np.isnan(v)]
    if len(all_aucs) == 0:
        return 0.0  # fallback if no AUC found
    return min(all_aucs)


#########################
# 8) plot_confusion_matrix
#########################
def plot_confusion_matrix(cm, label1, label2, output_dir):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {label1} vs {label2}")
    os.makedirs(output_dir, exist_ok=True)
    #plt.savefig(os.path.join(output_dir, f"ae_new_confusion_matrix_{label1}_vs_{label2}.pdf"), dpi=300, transparent=True)
    plt.close()


#########################
# 9) plot_single_aucs_heatmap
#########################
def plot_single_aucs_heatmap(
    single_means,
    single_stds,
    file_ids,
    backbones,
    output_dir,
    vmin=0.5,
    vmax=1.0
):
    """
    Plots a heatmap for single-file AUCs.
    Rows = file IDs, Columns = backbones.
    - single_means: 2D (n_files x n_backbones)
    - single_stds: 2D (n_files x n_backbones)
    """
    n_files, n_backbones = single_means.shape

    # Build custom annotation array: "mean\n(±std)"
    annot = np.empty_like(single_means, dtype=object)
    for i in range(n_files):
        for j in range(n_backbones):
            mean_val = single_means[i, j]
            std_val  = single_stds[i, j]
            if np.isnan(mean_val):
                annot[i, j] = ""
            elif np.isnan(std_val) or std_val == 0:
                annot[i, j] = f"{mean_val:.3f}"
            else:
                annot[i, j] = f"{mean_val:.3f}\n± {std_val:.3f}"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        single_means,
        annot=annot,
        fmt="",
        cmap="Reds",
        xticklabels=[b.upper() for b in backbones],
        yticklabels=[str(fid) for fid in file_ids],
        cbar_kws={'label': 'AUC'},
        annot_kws={"size": 10, "weight": "bold"},
        vmin=vmin,
        vmax=vmax
    )
    plt.title("Single-File AUC Heatmap", fontsize=16, weight="bold")
    plt.xlabel("Backbone", fontsize=12, weight="bold")
    plt.ylabel("File ID", fontsize=12, weight="bold")
    plt.xticks(fontsize=10, rotation=45, weight="bold")
    plt.yticks(fontsize=10, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "fus_status_new2_heatmap_single_files.pdf"), dpi=300, transparent=True)
    plt.close()


#########################
# 10) plot_triangle_heatmap_with_light_grey_diag
#########################
import matplotlib.colors as mcolors

import matplotlib.colors as mcolors

def plot_triangle_heatmap_with_light_grey_diag(
    mean_matrix,
    std_matrix,
    file_ids,
    backbone,
    output_dir,
    vmin=0.5,
    vmax=1.0
):
    """
    Plots the lower‑triangular heat‑map of `mean_matrix` (Reds) and shows
    the diagonal as light grey squares.  Lower‑triangle cells are annotated
    with either:
        • ""                     (if mean is NaN)
        • "µ"                    (if std is 0 or NaN)
        • "µ\n± σ"               (otherwise)
    """
    # ─── clean NaNs / sentinel values ────────────────────────────────────
    mean_matrix = np.where(mean_matrix == -1, np.nan, mean_matrix)
    std_matrix  = np.where(std_matrix  == -1, np.nan, std_matrix)
    n = mean_matrix.shape[0]

    # ─── build annotation array ─────────────────────────────────────────
    annot = np.empty_like(mean_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j or i < j:          # diag or upper‑tri → blank
                annot[i, j] = ""
            else:                        # lower‑triangle
                val_m = mean_matrix[i, j]
                val_s = std_matrix[i, j]
                if np.isnan(val_m):
                    annot[i, j] = ""
                elif np.isnan(val_s) or val_s == 0:
                    annot[i, j] = f"{val_m:.3f}"
                else:
                    annot[i, j] = f"{val_m:.3f}\n± {val_s:.3f}"

    labels = [str(fid) for fid in file_ids]

    # ─── masks for upper‑tri and diag ───────────────────────────────────
    lower_tri_mask = np.triu(np.ones_like(mean_matrix, dtype=bool), k=1)
    diag_mask      = ~np.eye(n, dtype=bool)

    # ─── 1) plot lower triangle ─────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        mean_matrix,
        mask=lower_tri_mask,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt="",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'AUC'},
        annot_kws={"size": 9, "weight": "bold"},
        alpha=1.0,
        zorder=1
    )

    # ─── 2) overlay light‑grey diagonal ─────────────────────────────────
    diag_vals = np.diag(mean_matrix).copy()
    np.fill_diagonal(mean_matrix, 999.0)          # sentinel
    light_grey_cmap = mcolors.ListedColormap(["#DDDDDD"])
    norm = mcolors.Normalize(vmin=998.5, vmax=999.5)

    sns.heatmap(
        mean_matrix,
        mask=diag_mask,
        cmap=light_grey_cmap,
        norm=norm,
        cbar=False,
        annot=False,
        alpha=1.0,
        zorder=2
    )
    # restore true diag
    for i in range(n):
        mean_matrix[i, i] = diag_vals[i]

    # ─── final cosmetics / save ─────────────────────────────────────────
    plt.title(f"Combination AUC for {backbone.upper()}",
              fontsize=14, weight="bold")
    plt.xlabel("Field of View", fontsize=12, weight="bold")
    plt.ylabel("Field of View", fontsize=12, weight="bold")
    plt.xticks(fontsize=10, rotation=45, weight="bold")
    plt.yticks(fontsize=10, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            output_dir,
            f"fus_status_ae_triangle_heatmap_lightgrey_diag_{backbone}.pdf"
        ),
        dpi=300,
        transparent=True
    )
    plt.close()



#########################
# 11) plot_boxplot_and_compute_pvals
#########################
def plot_boxplot_and_compute_pvals(
    single_aucs_distribution,
    backbone,
    file_ids,
    output_dir
):
    """
    1) Creates a Pandas DataFrame of all per-fold AUCs for the given backbone,
       with columns ['FileID', 'AUC'].
    2) Plots a boxplot with x=FileID, y=AUC.
    3) Computes pairwise t-test p-values for each (FileID1, FileID2) pair.
    """

    # Build the DataFrame
    rows = []
    for fid in file_ids:
        fold_aucs = single_aucs_distribution.get((backbone, fid), [])
        for val in fold_aucs:
            rows.append({"FileID": fid, "AUC": val})
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[Warning] No AUC distribution data found for backbone={backbone}. Skipping box plot.")
        return

    # 1) Boxplot
    plt.figure(figsize=(6,5))
    sns.boxplot(data=df, x="FileID", y="AUC", color="lightblue")
    sns.stripplot(data=df, x="FileID", y="AUC", color="black", alpha=0.7, jitter=True)
    plt.title(f"{backbone.upper()} Single-File AUC Distribution", fontsize=14, weight="bold")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Field of View (File ID)", fontsize=12, weight="bold")
    plt.ylabel("AUC", fontsize=12, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"fus_status_boxplot_single_{backbone}.pdf"), dpi=300, transparent=True)
    plt.close()

    # 2) Compute pairwise p-values
    print(f"\n=== Pairwise p-values for {backbone.upper()} ===")
    for i in range(len(file_ids)):
        for j in range(i+1, len(file_ids)):
            fid1 = file_ids[i]
            fid2 = file_ids[j]
            dist1 = df.loc[df["FileID"] == fid1, "AUC"].values
            dist2 = df.loc[df["FileID"] == fid2, "AUC"].values
            # Basic two-sided t-test
            tstat, pval = stats.ttest_ind(dist1, dist2, equal_var=False)
            print(f"  Compare FOV {fid1} vs {fid2}: t={tstat:.3f}, p={pval:.4g}")


#########################
# 12) compare_backbones_for_fov
#########################
def compare_backbones_for_fov(
    single_aucs_distribution,
    backbones,
    fov=2,
    output_dir="./"
):
    """
    1) Collect all fold-level AUCs for the specified FOV=2 (or any given fov).
    2) Plot a bar chart (with SD error bars) comparing each backbone.
    3) Perform pairwise t-tests between each pair of backbones.
    """
    # Build a DataFrame with columns: ['Backbone', 'AUC']
    rows = []
    for backbone in backbones:
        dist = single_aucs_distribution.get((backbone, fov), [])
        for auc_val in dist:
            rows.append({"Backbone": backbone, "AUC": auc_val})
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[Warning] No fold-level data found for FOV={fov}. Skipping bar chart.")
        return

    # Compute means, stds per backbone
    agg_df = df.groupby("Backbone")["AUC"].agg(["mean", "std"]).reset_index()

    # 1) Bar chart
    plt.figure(figsize=(6, 5))
    # We'll do a simple bar plot with standard-deviation error bars
    # For more control, we can plot manually, but seaborn's barplot with ci="sd" is straightforward:
    sns.barplot(data=df, x="Backbone", y="AUC", ci="sd", color="lightblue", capsize=0.2, errcolor="black")
    sns.stripplot(data=df, x="Backbone", y="AUC", color="black", alpha=0.7, jitter=True)
    plt.title(f"Comparison of Backbones at FOV={fov}", fontsize=14, weight="bold")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Backbone", fontsize=12, weight="bold")
    plt.ylabel("AUC", fontsize=12, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"fus_status_barplot_backbones_FOV{fov}.pdf"), dpi=300, transparent=True)
    plt.close()

    # 2) Pairwise p-values between backbones
    print(f"\n=== Pairwise p-values for FOV={fov} across backbones ===")
    for i in range(len(backbones)):
        for j in range(i+1, len(backbones)):
            b1 = backbones[i]
            b2 = backbones[j]
            dist1 = df.loc[df["Backbone"] == b1, "AUC"].values
            dist2 = df.loc[df["Backbone"] == b2, "AUC"].values
            tstat, pval = stats.ttest_ind(dist1, dist2, equal_var=False)
            print(f"  Compare {b1} vs {b2}: t={tstat:.3f}, p={pval:.4g}")


# ───────────────────── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR = "./output_fusion_cv"
    BACKBONES  = ['uni', 'conch', 'ctranspath', 'inception']
    FILE_IDS   = [1, 2, 3, 4]

    # 1) single-file CV
    print("=== SINGLE-file CV ===")
    single_means, single_stds, single_dist = test_single_files(BACKBONES, FILE_IDS)

    # 2) combination CV
    print("\n=== COMBINATION CV ===")
    combo_dict, cm_dict, met_dict = {}, {}, {}
    for bb in BACKBONES:
        auc_mat, std_mat, cms, mets = test_combinations(bb, FILE_IDS)
        combo_dict[bb], cm_dict[bb], met_dict[bb] = (auc_mat, std_mat), cms, mets

    # 3-N) plotting code can stay exactly as in your previous version
    for backbone in BACKBONES:
        print(f"\n========== Combination testing for backbone: {backbone} ==========")
        auc_matrix, std_matrix, confusion_matrices, metrics_list = test_combinations(backbone, FILE_IDS)
        combo_auc_dict[backbone] = (auc_matrix, std_matrix)
        combo_confusion_dict[backbone] = confusion_matrices
        combo_metrics_dict[backbone] = metrics_list

    # 3) Determine the global min AUC (we can fix the max at 1.0)
    global_min = find_global_auc_min(single_aucs_matrix, combo_auc_dict)
    global_max = 1.0
    print(f"\n[Info] Global AUC range across singles + combos: [{global_min:.3f}, {global_max:.3f}]")

    # 4) Plot single-file AUC heatmap
    print("\n=== Plotting SINGLE-FILE heatmap ===")
    plot_single_aucs_heatmap(
        single_aucs_matrix,
        single_aucs_stds_matrix,
        FILE_IDS,
        BACKBONES,
        OUTPUT_DIR,
        vmin=global_min,
        vmax=global_max
    )

    # 5) (Optional) Boxplots for each backbone across FOV=1..4
    print("\n=== Plotting BOX PLOTS + computing P-VALUES for each backbone (single-file) ===")
    for backbone in BACKBONES:
        plot_boxplot_and_compute_pvals(
            single_aucs_distribution,
            backbone,
            FILE_IDS,
            OUTPUT_DIR
        )

    # 6) Bar chart + significance across backbones at **FOV=2** 
    print("\n=== Comparing BACKBONES at FOV=2 ===")
    compare_backbones_for_fov(
        single_aucs_distribution,
        BACKBONES,
        fov=2,
        output_dir=OUTPUT_DIR
    )

    # 7) Plot combination-file heatmaps
    print("\n=== Plotting COMBINATION-FILE heatmaps ===")
    for backbone in BACKBONES:
        mean_mat, std_mat = combo_auc_dict[backbone]
        plot_triangle_heatmap_with_light_grey_diag(
            mean_mat,
            std_mat,
            FILE_IDS,
            backbone,
            OUTPUT_DIR,
            vmin=global_min,
            vmax=global_max
        )

        # Also print confusion matrices and metrics
        confusion_matrices = combo_confusion_dict[backbone]
        metrics_list = combo_metrics_dict[backbone]
        for (label1, label2, cm), (_, _, metrics) in zip(confusion_matrices, metrics_list):
            if cm is not None:
                plot_confusion_matrix(cm, label1, label2, OUTPUT_DIR)
            print(f"\nMetrics for {label1} vs {label2}:")
            for mkey, mval in metrics.items():
                if pd.notnull(mval):
                    print(f"  {mkey}: {mval:.3f}")
                else:
                    print(f"  {mkey}: None")
