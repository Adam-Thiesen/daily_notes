#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────── paths & settings ──────────────────────────────
H5AD          = "xc_tma_combined.h5ad"        # ← edit if needed
CL_KEY        = "leiden_1.0"                  # clustering column
N_TOP         = 10                            # genes per cluster
LAYER         = "counts"                      # raw counts are nicest for dotplots
OUTDIR        = Path("dotplots_leiden")       # one file per cluster
OUTDIR.mkdir(exist_ok=True, parents=True)

# ───────────────────────── load AnnData ──────────────────────────────────
adata = sc.read_h5ad(H5AD)
print(f"{adata.n_obs:,} cells  ×  {adata.n_vars:,} genes  loaded.")

# 1. (Re-)run rank_genes_groups on log-normalised data
rank_key = f"rank_{CL_KEY}"
if rank_key not in adata.uns:
    sc.tl.rank_genes_groups(
        adata,
        groupby     = CL_KEY,
        use_raw     = False,
        layer       = None,
        method      = "wilcoxon",
        n_genes     = N_TOP,
        key_added   = rank_key
    )

# 2. Collect top-N genes per cluster
genes_per_cluster = {}
for cl in adata.obs[CL_KEY].cat.categories:
    df = sc.get.rank_genes_groups_df(
        adata,
        group       = cl,
        key         = rank_key,
        log2fc_min  = None,
        pval_cutoff = None
    ).head(N_TOP)
    genes_per_cluster[cl] = [g for g in df["names"] if g in adata.var_names]

print("Example:", {k: v for k, v in list(genes_per_cluster.items())[:2]})

# 3. One dot-plot per cluster, rotating x labels and using bbox_inches
for cl, genes in genes_per_cluster.items():
    if not genes:
        continue

    # put the current cluster first
    order = [cl] + [c for c in adata.obs[CL_KEY].cat.categories if c != cl]
    adata.obs["__tmp_order"] = adata.obs[CL_KEY].cat.reorder_categories(order, ordered=True)

    fig_path = OUTDIR / f"dot_leiden_{cl}.png"
    sc.pl.dotplot(
        adata,
        var_names      = genes,
        groupby        = "__tmp_order",
        layer          = LAYER,
        standard_scale = "var",
        color_map      = "RdBu_r",
        dot_max        = 0.5,
        figsize        = (0.8 * len(genes) + 4, 8),
        dendrogram     = False,
        show           = False
    )

    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # save with extra margin at bottom for rotated labels
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

adata.obs.drop(columns="__tmp_order", inplace=True)
print("✓ dot-plots written to", OUTDIR)
