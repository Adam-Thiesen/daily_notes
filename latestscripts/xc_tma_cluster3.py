#!/usr/bin/env python3
"""
xc_tma_cluster.py  –  Combine per-core Xenium .h5ad files, run light QC,
normalise, cluster *using every gene*, and save results.

Typical usage
-------------
python xc_tma_cluster.py \
    --core-dir per_core_h5ad \
    --exclude 501 601 \
    --out-h5ad xc_tma_combined.h5ad \
    --umap-pdf xc_tma_umap.pdf
"""

import argparse, sys, warnings
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np

# ───────────────────────────── QC PARAMETERS ──────────────────────────────
MIN_COUNTS          = 25
MAX_COUNTS          = 4e4
MIN_GENES           = 25
MAX_PCT_MT          = 25.0        # %
MAX_BG_FRACTION     = 0.20        # control_probe + unassigned_codeword
AREA_QUANTILES      = (0.01, 0.99)
DOUBLETS_THRESHOLD  = 0.25        # Scrublet score cut-off

# ───────────────────────────── CLI ────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Xenium TMA: concatenate cores, light-QC, full-gene clustering"
    )
    p.add_argument("--core-dir", type=Path, default="per_core_h5ad",
                   help="Directory containing per-core *.h5ad files")
    p.add_argument("--exclude", type=int, nargs="*", default=[],
                   help="Core IDs to skip")
    p.add_argument("--out-h5ad", default="xc_tma_combined_5.h5ad",
                   help="Output combined AnnData filename")
    p.add_argument("--umap-pdf", default="xc_tma_umap_5.pdf",
                   help="Filename for UMAP PDF")
    p.add_argument("--figdir", default="figures",
                   help="Folder for Scanpy figures")
    p.add_argument("--n-hvg", type=int, default=3000,
                   help="Number of HVGs to *tag* (not used for clustering)")
    p.add_argument("--random-seed", type=int, default=0,
                   help="Seed for reproducible neighbours/UMAP")
    p.add_argument("--no-doublet", action="store_true",
                   help="Skip Scrublet doublet detection")
    return p.parse_args()

# ───────────────────────── helpers ────────────────────────────────────────
def qc_per_core(ad, cid, no_doublet=False):
    """Add QC metrics & filter obvious low-quality cells in one AnnData."""
    # mark mitochondrial genes
    ad.var["mt"] = ad.var_names.str.startswith(("mt-", "MT-"))

    sc.pp.calculate_qc_metrics(
        ad,
        qc_vars={"mt": ad.var["mt"]},
        percent_top=None,
        layer="counts",          # raw layer set later
        inplace=True,
    )

    # ------------------------------------------------------------------ #
    # Background fraction – tolerate missing columns                      #
    # ------------------------------------------------------------------ #
    ctrl = ad.obs.get(
        "control_probe_counts",
        pd.Series(0, index=ad.obs.index, dtype=float)
    )
    unassigned = ad.obs.get(
        "unassigned_codeword_counts",
        pd.Series(0, index=ad.obs.index, dtype=float)
    )
    bg_fraction = (ctrl + unassigned) / ad.obs.total_counts.clip(lower=1)
    ad.obs["bg_fraction"] = bg_fraction

    # ------------------------------------------------------------------ #
    # Core QC mask                                                       #
    # ------------------------------------------------------------------ #
    qc_mask = (
        (ad.obs.total_counts      >= MIN_COUNTS) &
        (ad.obs.total_counts      <= MAX_COUNTS) &
        (ad.obs.n_genes_by_counts >= MIN_GENES)  &
        (ad.obs.pct_counts_mt     <= MAX_PCT_MT) &
        (bg_fraction              <= MAX_BG_FRACTION)
    )

    # drop extreme cell areas (only if metadata present)
    if "cell_area" in ad.obs.columns:
        lo, hi = ad.obs.cell_area.quantile(AREA_QUANTILES).values
        qc_mask &= ad.obs.cell_area.between(lo, hi)

    # optional doublet detection
    if not no_doublet:
        try:
            import scrublet as scr
            scrub = scr.Scrublet(ad.layers["counts"])
            doublet_scores, _ = scrub.scrub_doublets(min_counts=2,
                                                     random_state=0)
            ad.obs["doublet_score"] = doublet_scores
            qc_mask &= ad.obs.doublet_score < DOUBLETS_THRESHOLD
        except ModuleNotFoundError:
            warnings.warn("Scrublet not installed – skipping doublet detection.")
            ad.obs["doublet_score"] = np.nan
    else:
        ad.obs["doublet_score"] = np.nan

    n_before = ad.n_obs
    ad = ad[qc_mask].copy()
    print(f"• QC core {cid:>3}: kept {ad.n_obs}/{n_before} cells")
    return ad

# ───────────────────────── main ───────────────────────────────────────────
def main() -> None:
    args = parse_args()
    sc.settings.verbosity = 2
    sc.settings.figdir    = args.figdir
    Path(args.figdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load per-core files                                                #
    # ------------------------------------------------------------------ #
    adatas = []
    for fn in sorted(args.core_dir.glob("core_*.h5ad")):
        ad = sc.read_h5ad(fn)
        cid = int(ad.obs["core_ID"].iloc[0])
        if cid in args.exclude:
            print(f"• skipped control core {cid} ({fn.name})")
            continue

        # snapshot raw counts for QC & downstream DE
        ad.layers["counts"] = ad.X

        # per-core QC
        ad = qc_per_core(ad, cid, no_doublet=args.no_doublet)
        if ad.n_obs:                      # skip empty cores
            adatas.append(ad)

    if not adatas:
        sys.exit("No usable cores – check --core-dir / --exclude")

    print(f"Loaded {len(adatas)} cores → total cells:",
          sum(a.n_obs for a in adatas))

    # ------------------------------------------------------------------ #
    # Concatenate & basic preprocessing                                  #
    # ------------------------------------------------------------------ #
    adata = sc.concat(adatas, join="outer",
                      label="core_ID", index_unique=None)

    # Drop genes never expressed (frees RAM)
    sc.pp.filter_genes(adata, min_cells=10)

    # Tag HVGs (not used for clustering but useful downstream)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=args.n_hvg,
        batch_key="core_ID",
        layer="counts",
    )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False)   # keeps sparsity

    # ------------------------------------------------------------------ #
    # PCA → neighbours → Leiden → UMAP                                   #
    # ------------------------------------------------------------------ #
    sc.tl.pca(
        adata,
        svd_solver="randomized",
        n_comps=40,
        zero_center=False,
        random_state=args.random_seed,
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=15,
        n_pcs=40,
        random_state=args.random_seed,
    )
    sc.tl.leiden(
        adata,
        resolution = 0.1,
        key_added="leiden_1.0",
        random_state=args.random_seed,
    )
    sc.tl.umap(adata, random_state=args.random_seed)

    # ------------------------------------------------------------------ #
    # Plots                                                             #
    # ------------------------------------------------------------------ #
    sc.pl.umap(
        adata,
        color=["leiden_1.0", "core_ID"],
        save="_quick.png",
        show=False,
    )
    sc.pl.umap(
        adata,
        color=["leiden_1.0", "core_ID"],
        save=args.umap_pdf,
        show=False,
    )

    sc.pl.violin(
        adata,
        ["total_counts", "n_genes_by_counts", "pct_counts_mt",
         "bg_fraction", "doublet_score"],
        groupby="core_ID",
        rotation=90,
        save="_qc_violin_1.png",
        show=False,
    )

    # ------------------------------------------------------------------ #
    # Cluster × core counts                                              #
    # ------------------------------------------------------------------ #
    (
        adata.obs.groupby(["core_ID", "leiden_1.0"])
        .size()
        .rename("n_cells")
        .reset_index()
        .to_csv("cluster_counts_per_core_1.csv", index=False)
    )

    # ------------------------------------------------------------------ #
    # Save                                                               #
    # ------------------------------------------------------------------ #
    adata.write(args.out_h5ad)
    print("✓ combined AnnData →", args.out_h5ad)
    print("✓ cluster counts   → cluster_counts_per_core_1.csv")
    print("✓ figures          →", args.figdir)

# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
