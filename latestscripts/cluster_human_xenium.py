#!/usr/bin/env python3
"""
xenium5k_qc_cluster.py
- Discover multiple Xenium runs under a root directory
- Apply light QC compatible with Xenium (bg fraction, MT%, area trimming, optional Scrublet)
- Normalize/log, (optionally) restrict to HVGs, PCA → neighbors → Leiden
- UMAP with optional fit-on-subsample + transform for speed
- Save combined .h5ad and summaries

Example:
python xenium5k_qc_cluster.py \
  --xenium-root /path/to/all_xenium_runs \
  --out-h5ad xc_5k_all_runs.h5ad \
  --cluster-on hvg \
  --n-hvg 3000 \
  --n-pcs 50 \
  --n-neighbors 30 \
  --resolution 0.3 \
  --umap-on-sample 150000 \
  --figdir figures_all \
  --write-per-experiment
"""

import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

# ───────────────────────────── QC PARAMETERS (defaults) ───────────────────
MIN_COUNTS          = 25
MAX_COUNTS          = 4e4
MIN_GENES           = 25
MAX_PCT_MT          = 25.0        # %
MAX_BG_FRACTION     = 0.20        # (control + unassigned + deprecated) / total_counts
AREA_QUANTILES      = (0.01, 0.99)
DOUBLETS_THRESHOLD  = 0.25        # Scrublet score cutoff
GENE_MIN_CELLS      = 10          # pre-filter genes to reduce memory

# ───────────────────────────── CLI ────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC & clustering across multiple Xenium 5k runs")
    p.add_argument("--xenium-root", type=Path, required=True,
                   help="Directory with one subfolder per Xenium experiment/run")
    p.add_argument("--include", nargs="*", default=None,
                   help="Optional list of run folder names to include (default: auto-detect all)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Run folder names to skip")
    p.add_argument("--out-h5ad", default="xc_5k_all_runs.h5ad",
                   help="Output AnnData filename")
    p.add_argument("--figdir", default="figures",
                   help="Folder for Scanpy figures")
    p.add_argument("--umap-suffix", default="_x5k",
                   help="Suffix for saved UMAP files, e.g., _x5k")
    p.add_argument("--n-hvg", type=int, default=3000,
                   help="Number of HVGs to tag (and optionally use for clustering)")
    p.add_argument("--cluster-on", choices=["hvg","all"], default="hvg",
                   help="Use only HVGs for PCA/neighbors (hvg) or all genes (all). Default: hvg")
    p.add_argument("--n-pcs", type=int, default=50,
                   help="Number of PCs for neighbors/UMAP")
    p.add_argument("--n-neighbors", type=int, default=30,
                   help="k for KNN graph")
    p.add_argument("--resolution", type=float, default=0.3,
                   help="Leiden resolution")
    p.add_argument("--random-seed", type=int, default=0,
                   help="Seed for reproducibility")
    p.add_argument("--no-doublet", action="store_true",
                   help="Skip Scrublet doublet detection (recommended for very large data)")
    p.add_argument("--skip-violin", action="store_true",
                   help="Skip QC violin plots")
    p.add_argument("--gene-min-cells", type=int, default=GENE_MIN_CELLS,
                   help="Drop genes detected in fewer than this many cells before PCA (default 10)")
    p.add_argument("--neighbors-metric", default="cosine",
                   help="Distance metric for neighbors (default cosine)")
    p.add_argument("--neighbors-method", default="umap",
                   help="Neighbors method (scanpy passes to pynndescent if 'umap')")
    p.add_argument("--umap-on-sample", type=int, default=0,
                   help="If >0, fit UMAP on a random subset of this many cells and transform the rest (uses umap-learn)")
    p.add_argument("--save-intermediate", action="store_true",
                   help="Save a pre-clustering .h5ad after QC+norm (heavy datasets)")
    p.add_argument("--write-per-experiment", action="store_true",
                   help="Write per-experiment .h5ad and CSV (cell_id, cluster, x, y).")
    return p.parse_args()

# ───────────────────────── helpers ────────────────────────────────────────
def _mark_mt_genes(var_names: pd.Index) -> pd.Series:
    v = var_names.astype(str).str.upper()
    return v.str.startswith("MT-") | v.str.match(r"^MT[._-]")

def _ensure_counts_layer_is_raw(ad):
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X

def _coerce_float32_layer(ad, layer_key):
    try:
        X = ad.layers.get(layer_key, None)
        if X is None:
            return
        if getattr(X, "dtype", None) != np.float32:
            if issparse(X):
                ad.layers[layer_key] = X.astype(np.float32)
            else:
                ad.layers[layer_key] = np.asarray(X, dtype=np.float32)
    except Exception:
        pass

def _discover_runs(root: Path):
    """
    Recursively find Xenium run directories under `root`.
    A 'run directory' is any directory that contains either:
      - cell_feature_matrix.h5   (Xenium matrix)
      - *.h5ad                   (pre-made AnnData for that run)
    Returns a sorted list of Path objects (the directories).
    """
    run_dirs = set()
    # Xenium matrices
    for h5 in root.rglob("cell_feature_matrix.h5"):
        run_dirs.add(h5.parent)
    # Pre-made AnnData (if you sometimes save a per-run h5ad)
    for h5ad in root.rglob("*.h5ad"):
        run_dirs.add(h5ad.parent)
    return sorted(run_dirs)


def _load_one_run(run_dir: Path, seed=0):
    """
    Load one Xenium run from a directory:
    - Prefer any *.h5ad in the folder (assumed raw counts in X or counts layer)
    - Else, read 10x Xenium matrix H5 (cell_feature_matrix.h5) via scanpy.read_10x_h5
      and optionally merge cells.csv(.gz) into obs if present.
    Returns AnnData with obs['experiment_id'] set and (if available) obsm['spatial'].
    """
    exp_id = run_dir.name

    # Prefer a single h5ad if present
    h5ads = sorted(run_dir.glob("*.h5ad"))
    if h5ads:
        ad = sc.read_h5ad(h5ads[0])
    else:
        h5 = run_dir / "cell_feature_matrix.h5"
        if not h5.exists():
            raise FileNotFoundError(f"No cell_feature_matrix.h5 or *.h5ad in {run_dir}")
        ad = sc.read_10x_h5(h5)  # counts in X (sparse)
        # Try to merge cells.csv(.gz) for QC columns (areas, totals, etc.)
        cells_csv = None
        for pat in ("cells.csv.gz", "cells.csv"):
            f = run_dir / pat
            if f.exists():
                cells_csv = f
                break
        if cells_csv is not None:
            try:
                df = pd.read_csv(cells_csv)
                # heuristic: if there is an ID column, index by it
                cand_idx = None
                for k in ("cell_id", "cell_ID", "ID", "barcode"):
                    if k in df.columns:
                        cand_idx = k
                        break
                if cand_idx is not None:
                    df = df.set_index(cand_idx, drop=False)
                # align on obs_names if possible
                common = ad.obs_names.intersection(df.index.astype(str))
                if len(common) > 0:
                    ad.obs = ad.obs.join(df.loc[common], how="left")
                    # standardize spatial coords if present
                    for xk, yk in [("x_centroid","y_centroid"), ("x","y"), ("x_global","y_global")]:
                        if xk in ad.obs and yk in ad.obs:
                            xy = ad.obs[[xk, yk]].astype(float).to_numpy()
                            ad.obsm["spatial"] = xy
                            ad.obs["x"] = ad.obsm["spatial"][:,0]
                            ad.obs["y"] = ad.obsm["spatial"][:,1]
                            break
            except Exception as e:
                warnings.warn(f"Could not merge cells.csv for {run_dir}: {e}")

    ad.obs["experiment_id"] = exp_id
    # Optional: record the run path for traceability
    ad.obs["experiment_path"] = str(run_dir)
    return ad

def qc_per_subset(ad, no_doublet=False, seed=0, run_label=""):
    """
    Add QC metrics & filter obvious low-quality cells in one AnnData subset (one run),
    using layers['counts'] as the source of truth.
    """
    _ensure_counts_layer_is_raw(ad)

    ad.var["mt"] = _mark_mt_genes(ad.var_names)
    sc.pp.calculate_qc_metrics(
        ad,
        qc_vars={"mt": ad.var["mt"]},
        percent_top=None,
        layer="counts",
        inplace=True,
    )

    # background fraction
    ctrl       = ad.obs.get("control_probe_counts",        pd.Series(0, index=ad.obs.index, dtype=float))
    unassigned = ad.obs.get("unassigned_codeword_counts",  pd.Series(0, index=ad.obs.index, dtype=float))
    deprecated = ad.obs.get("deprecated_codeword_counts",  pd.Series(0, index=ad.obs.index, dtype=float))

    if "total_counts" not in ad.obs:
        try:
            ad.obs["total_counts"] = np.asarray(ad.layers["counts"].sum(axis=1)).ravel()
        except Exception:
            warnings.warn(f"{run_label}: could not compute total_counts; filling 1s to avoid div-by-zero")
            ad.obs["total_counts"] = 1.0

    bg_fraction = (pd.Series(ctrl, index=ad.obs.index, dtype=float)
                   + pd.Series(unassigned, index=ad.obs.index, dtype=float)
                   + pd.Series(deprecated, index=ad.obs.index, dtype=float)) / np.clip(ad.obs["total_counts"].values, 1, None)
    ad.obs["bg_fraction"] = np.asarray(bg_fraction, dtype=float)

    qc_mask = (
        (ad.obs.total_counts      >= MIN_COUNTS) &
        (ad.obs.total_counts      <= MAX_COUNTS) &
        (ad.obs.n_genes_by_counts >= MIN_GENES)  &
        (ad.obs.pct_counts_mt     <= MAX_PCT_MT) &
        (ad.obs.bg_fraction       <= MAX_BG_FRACTION)
    )

    if "cell_area" in ad.obs.columns:
        lo, hi = ad.obs["cell_area"].quantile(AREA_QUANTILES).values
        qc_mask &= ad.obs["cell_area"].between(lo, hi)

    if not no_doublet:
        try:
            import scrublet as scr
            scrub = scr.Scrublet(ad.layers["counts"], random_state=seed)
            doublet_scores, _ = scrub.scrub_doublets(min_counts=2, random_state=seed)
            ad.obs["doublet_score"] = doublet_scores
            qc_mask &= ad.obs["doublet_score"] < DOUBLETS_THRESHOLD
        except ModuleNotFoundError:
            warnings.warn("Scrublet not installed – skipping doublet detection.")
            ad.obs["doublet_score"] = np.nan
    else:
        ad.obs["doublet_score"] = np.nan

    n_before = ad.n_obs
    ad = ad[qc_mask].copy()
    print(f"• QC {run_label:>15}: kept {ad.n_obs}/{n_before} cells")
    return ad

# ───────────────────────── main ───────────────────────────────────────────
def main():
    args = parse_args()

    sc.settings.verbosity = 2
    sc.settings.figdir    = args.figdir
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    np.random.seed(args.random_seed)

    # discover runs
    all_runs = _discover_runs(args.xenium_root)
    if not all_runs:
        sys.exit(f"No runs found under {args.xenium_root}")

    if args.include is not None:
        targets = set(args.include)
        all_runs = [p for p in all_runs if p.name in targets]
    if args.exclude:
        bad = set(args.exclude)
        all_runs = [p for p in all_runs if p.name not in bad]

    if not all_runs:
        sys.exit("No runs to process after include/exclude filters.")

    print("Discovered runs:")
    for p in all_runs:
        print("  -", p)

    # load and QC each run independently (keeps memory footprint bounded per run)
    adatas = []
    for run in all_runs:
      # exp_id = first directory name under the root (e.g., "section1")
      rel_parts = run.relative_to(args.xenium_root).parts
      exp_id = rel_parts[0] if len(rel_parts) > 0 else run.name
  
      ad = _load_one_run(run, seed=args.random_seed)
  
      # set experiment metadata and unique cell IDs before QC
      
      ad.obs_names = ad.obs_names.map(lambda s: f"{exp_id}__{s}")
  
      _ensure_counts_layer_is_raw(ad)
      ad = qc_per_subset(ad, no_doublet=args.no_doublet, seed=args.random_seed, run_label=exp_id)
  
      if ad.n_obs > 0:
          adatas.append(ad)


    if not adatas:
        sys.exit("All runs empty after QC.")

    print(f"Loaded {len(adatas)} runs → total cells:", sum(a.n_obs for a in adatas))

    # concat across runs
    adata = sc.concat(adatas, join="outer", index_unique=None)

    # coerce counts to float32 (helps some ops / memory)
    _coerce_float32_layer(adata, "counts")

    # ensure experiment_id is categorical & ordered for plotting
    if "experiment_id" in adata.obs:
        cats = [str(x) for x in sorted(pd.Series(adata.obs["experiment_id"]).astype(str).unique())]
        adata.obs["experiment_id"] = pd.Categorical(adata.obs["experiment_id"].astype(str),
                                                    categories=cats, ordered=True)

    # Pre-filter genes rarely expressed
    sc.pp.filter_genes(adata, min_cells=args.gene_min_cells)

    # Tag HVGs from raw counts (batch-aware by experiment)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=args.n_hvg,
        batch_key=("experiment_id" if "experiment_id" in adata.obs else None),
        layer="counts",
    )

    # Normalize & log from counts
    if "counts" not in adata.layers:
        raise RuntimeError("layers['counts'] not found; expected raw counts there.")
    adata.X = adata.layers["counts"].copy()
    try:
        sc.pp.normalize_total(adata, target_sum=1e4)
    except TypeError:
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # keep log1p copy + raw for plotting
    adata.layers["log1p"] = adata.X.copy()
    adata.raw = adata.copy()

    # Optionally restrict to HVGs for PCA/neighbors
    use_hvg = (args.cluster_on == "hvg") and ("highly_variable" in adata.var)
    if use_hvg:
        adata._inplace_subset_var(adata.var["highly_variable"].values)

    # Scale for PCA/neighbors (avoid centering for sparse safety)
    sc.pp.scale(adata, zero_center=False, max_value=None)

    if args.save_intermediate:
        mid = Path(args.out_h5ad).with_suffix("").as_posix() + "_postQCnorm.h5ad"
        print("Saving intermediate:", mid)
        adata.write(mid, compression="gzip")

    # PCA → neighbors → Leiden
    n_pcs = args.n_pcs
    sc.tl.pca(
        adata,
        svd_solver="randomized",
        n_comps=n_pcs,
        zero_center=False,
        random_state=args.random_seed,
    )
    try:
        sc.pp.neighbors(
            adata,
            n_neighbors=args.n_neighbors,
            n_pcs=n_pcs,
            random_state=args.random_seed,
            method=args.neighbors_method,   # 'umap' -> pynndescent if available
            metric=args.neighbors_metric,
        )
    except TypeError:
        # Older Scanpy may not accept 'method' or 'metric'
        sc.pp.neighbors(
            adata,
            n_neighbors=args.n_neighbors,
            n_pcs=n_pcs,
            random_state=args.random_seed,
        )

    leiden_key = f"leiden_{args.resolution:g}"
    sc.tl.leiden(
        adata,
        resolution=args.resolution,
        key_added=leiden_key,
        random_state=args.random_seed,
    )

    # UMAP: either standard scanpy or fit-on-subsample + transform
    if args.umap_on_sample > 0 and adata.n_obs > args.umap_on_sample:
        try:
            import umap
            print(f"Fitting UMAP on {args.umap_on_sample} subsampled cells and transforming the rest...")
            X_pca = adata.obsm["X_pca"]
            rng = np.random.default_rng(args.random_seed)
            idx = rng.choice(adata.n_obs, size=args.umap_on_sample, replace=False)
            mask = np.zeros(adata.n_obs, dtype=bool)
            mask[idx] = True

            reducer = umap.UMAP(
                n_neighbors=args.n_neighbors,
                n_components=2,
                min_dist=0.5,
                metric="euclidean",
                random_state=args.random_seed,
            )
            emb_fit = reducer.fit_transform(X_pca[mask])
            emb_all = np.zeros((adata.n_obs, 2), dtype=np.float32)
            emb_all[mask] = emb_fit
            emb_all[~mask] = reducer.transform(X_pca[~mask])
            adata.obsm["X_umap"] = emb_all
        except Exception as e:
            warnings.warn(f"Subsampled UMAP failed with error '{e}'. Falling back to scanpy.tl.umap.")
            sc.tl.umap(adata, random_state=args.random_seed)
    else:
        sc.tl.umap(adata, random_state=args.random_seed)

    # ── Plots ─────────────────────────────────────────────────────────────
    sc.pl.umap(adata, color=[leiden_key, "experiment_id"], save=args.umap_suffix + ".png", show=False)
    sc.pl.umap(adata, color=[leiden_key, "experiment_id"], save=args.umap_suffix + ".pdf", show=False)

    if not args.skip_violin:
        qc_cols = [c for c in ["total_counts","n_genes_by_counts","pct_counts_mt","bg_fraction","doublet_score"] if c in adata.obs]
        if qc_cols:
            sc.pl.violin(
                adata, qc_cols,
                groupby=("experiment_id" if "experiment_id" in adata.obs else leiden_key),
                rotation=90, save="_qc_violin_x5k.png", show=False
            )

    # per-experiment cluster counts
    gb_cols = ["experiment_id", leiden_key] if "experiment_id" in adata.obs else [leiden_key]
    counts = (
        adata.obs.groupby(gb_cols)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    counts.to_csv("cluster_counts_per_experiment.csv", index=False)

    # Optional: write per-experiment splits for easy overlays / native viewers
    if args.write_per_experiment and "experiment_id" in adata.obs:
        outbase = Path(args.out_h5ad).with_suffix("").as_posix()
        for exp in adata.obs["experiment_id"].cat.categories:
            ad_exp = adata[adata.obs["experiment_id"] == exp].copy()
            exp_fn = f"{outbase}__{exp}.h5ad"
            ad_exp.write(exp_fn, compression="gzip")
            cols = ["experiment_id", leiden_key]
            for c in ("x","y"):
                if c in ad_exp.obs: cols.append(c)
            df = ad_exp.obs[cols].copy()
            df.insert(0, "cell_id", ad_exp.obs_names)
            df.to_csv(f"{outbase}__{exp}__cells_clusters.csv", index=False)
            print("  • wrote", exp_fn, "and CSV of cell coordinates+clusters")

    # ── Save combined ─────────────────────────────────────────────────────
    adata.write(args.out_h5ad, compression="gzip")
    print("✓ combined AnnData →", args.out_h5ad)
    print("✓ cluster counts   → cluster_counts_per_experiment.csv")
    print("✓ figures          →", args.figdir)
    print("Layout summary:")
    print("  X                  = scaled working matrix (HVGs if --cluster-on hvg)")
    print("  layers['log1p']    = log1p-normalized (pre-scale)")
    print("  layers['counts']   = raw counts")
    print("  raw.X              = log1p-normalized (pre-scale) for plotting")
    print("  var['highly_variable'] tagged from counts (used if --cluster-on hvg)")
    print(f"  UMAP mode          = {'subsample-transform' if args.umap_on_sample>0 else 'standard'}")

if __name__ == "__main__":
    main()
