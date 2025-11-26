#!/usr/bin/env python3
# --- Paths (edit) ---
xeBundlePath = "/projects/chuang-lab/thiesa/SL250282-XR25010_0020938/xeniumranger/region1"
PATH_10X     = f"{xeBundlePath}/cell_feature_matrix.h5"          # Xenium 10x-style H5
PATH_P21     = "/flashscratch/thiesa/align2/cells-codex-trimmed_p21bins.parquet"
OUTDIR       = "/flashscratch/thiesa/align2/p21_DE"

# --- Imports ---
import os, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq

# --- Setup ---
os.makedirs(OUTDIR, exist_ok=True)
sc.settings.figdir = OUTDIR
sc.settings.set_figure_params(dpi=200)

# ---- 1) Build AnnData from Xenium H5 (counts) ----
ad = sc.read_10x_h5(PATH_10X)   # counts; ad.X is sparse CSR
ad.var_names_make_unique()
print(f"[info] loaded counts: n_cells={ad.n_obs}, n_genes={ad.n_vars}")

# ---- 2) Drop zero-count cells ----
if sp.issparse(ad.X):
    totals = np.asarray(ad.X.sum(axis=1)).ravel()
else:
    totals = ad.X.sum(axis=1)
nz_mask = totals > 0
if nz_mask.sum() < ad.n_obs:
    print(f"[info] dropping zero-count cells: {ad.n_obs - nz_mask.sum()} / {ad.n_obs}")
    ad = ad[nz_mask].copy()

# ---- 3) Align obs names to Xenium cell_id using cells.parquet if necessary ----
cells_df = pd.read_parquet(f"{xeBundlePath}/cells.parquet")  # 'cell_id', maybe 'barcode'
ad.obs_names = ad.obs_names.astype(str)

if "cell_id" in cells_df.columns:
    cells_df["cell_id"] = cells_df["cell_id"].astype(str)
    cand_barcode = [c for c in ["barcode", "cell_barcode", "barcode_seq"] if c in cells_df.columns]
    if cand_barcode and not ad.obs_names.isin(cells_df["cell_id"]).all():
        bc_col = cand_barcode[0]
        cells_df[bc_col] = cells_df[bc_col].astype(str)
        b2id = dict(zip(cells_df[bc_col], cells_df["cell_id"]))
        mappable = ad.obs_names.isin(b2id.keys())
        if mappable.any():
            ad = ad[mappable].copy()
            ad.obs_names = pd.Index([b2id[b] for b in ad.obs_names], name="cell_id")
            print(f"[info] mapped {mappable.sum()} barcodes → cell_id via '{bc_col}'")
        else:
            ad.obs_names.name = "cell_id"
            print("[warn] could not map via barcode; assuming obs_names are cell_id")
    else:
        ad.obs_names.name = "cell_id"
else:
    ad.obs_names.name = "cell_id"
    print("[warn] 'cell_id' not found in cells.parquet; proceeding with current obs_names")

# ---- 4) Light normalization (log1p) + set raw ----
sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
ad.layers["log1p"] = ad.X.copy()
ad.raw = ad.copy()  # USE_RAW=True will use this
print("[info] created layers: 'log1p' and set ad.raw")

# ---- 5) Load p21 bins, align IDs (barcode↔cell_id, label_id→cell_id if needed) ----
p21 = pd.read_parquet(PATH_P21)
ad.obs_names = ad.obs_names.astype(str)
p21.index    = p21.index.astype(str)
print("[diag] initial overlap:", ad.obs_names.intersection(p21.index).size)

# 5a) barcode→cell_id (if obs_names are barcodes; already tried above)
if ad.obs_names.intersection(p21.index).size == 0:
    cells_path = Path(xeBundlePath) / "cells.parquet"
    if cells_path.exists():
        cols = [c for c in ["cell_id","barcode","cell_barcode","barcode_seq"] if pq.read_table(cells_path).schema.get_field_index(c) != -1]
        cells = pd.read_parquet(cells_path, columns=cols)
        for c in cells.columns: cells[c] = cells[c].astype(str)
        cand_bc = [c for c in ["barcode","cell_barcode","barcode_seq"] if c in cells.columns]
        if "cell_id" in cells.columns and cand_bc:
            bc_col = cand_bc[0]
            b2id = dict(zip(cells[bc_col], cells["cell_id"]))
            mappable = ad.obs_names.isin(b2id.keys())
            if mappable.any():
                ad = ad[mappable].copy()
                ad.obs_names = pd.Index([b2id[b] for b in ad.obs_names], name="cell_id")
                print(f"[fix] mapped {mappable.sum()} barcodes → cell_id via '{bc_col}'")
        else:
            print("[warn] cells.parquet missing barcode column; skipping barcode→cell_id mapping")

# 5b) If still no overlap, remap p21 index from label_id → cell_id via boundaries
if ad.obs_names.intersection(p21.index).size == 0:
    bounds_path = Path(xeBundlePath) / "cell_boundaries.parquet"
    if bounds_path.exists():
        schema = pq.read_schema(bounds_path)
        needed = [c for c in ["label_id","cell_id"] if c in schema.names]
        if set(["label_id","cell_id"]).issubset(needed):
            bdf = pd.read_parquet(bounds_path, columns=needed)
            bdf = bdf.drop_duplicates(subset=["label_id"])[["label_id","cell_id"]].astype(str)
            lab2cid = dict(zip(bdf["label_id"], bdf["cell_id"]))
            known = p21.index.isin(lab2cid.keys())
            if known.any():
                mapped_index = p21.index.where(~known, pd.Index([lab2cid[x] for x in p21.index[known]]))
                p21 = p21.set_index(mapped_index.astype(str))
                print(f"[fix] remapped p21 index label_id → cell_id using boundaries ({known.sum()} ids)")
            else:
                print("[warn] boundaries present but no p21 indices matched label_id")
        else:
            print("[warn] cell_boundaries.parquet lacks both 'label_id' and 'cell_id'; cannot remap")

overlap = ad.obs_names.intersection(p21.index)
print("[diag] final overlap:", overlap.size)
if overlap.size == 0:
    print("[peek] ad.obs_names head:", list(ad.obs_names[:5]))
    print("[peek] p21 index head:", list(p21.index[:5]))
    raise ValueError("Still no overlap after mapping attempts. Inspect the [peek] IDs above.")

# ---- 6) Join + subset to high/low ----
keep = overlap
ad   = ad[keep].copy()
p21  = p21.loc[keep]

ad.obs["p21_bin"] = pd.Categorical(p21["p21_bin"], categories=["low", "mid", "high"])
vc_all = ad.obs["p21_bin"].value_counts(dropna=False)
print(f"[diag] p21_bin counts (all):\n{vc_all}")

ad_hl = ad[ad.obs["p21_bin"].isin(["low", "high"])].copy()
ad_hl.obs["p21_bin"] = pd.Categorical(ad_hl.obs["p21_bin"], categories=["low", "high"], ordered=True)
vc_hl = ad_hl.obs["p21_bin"].value_counts()
print(f"[info] using high/low only: {vc_hl.to_dict()}")
if ad_hl.n_obs == 0 or vc_hl.min() == 0:
    raise ValueError("High/low bins are empty or one-sided after join. Verify p21 binning and ID overlap.")

# ---- 7) DE (high vs low) ----
USE_RAW     = True          # use ad.raw (the log1p-normalized matrix above)
LAYER_PLOT  = "log1p"       # used only if USE_RAW=False
RANK_METHOD = "wilcoxon"
TOP_N       = 20
DOT_MAX     = 0.5

sc.tl.rank_genes_groups(
    ad_hl,
    groupby="p21_bin",
    groups=["high"], reference="low",
    method=RANK_METHOD,
    use_raw=USE_RAW,
    layer=None if USE_RAW else LAYER_PLOT,
    pts=True,
    key_added="p21_wilcoxon",
)

# ---- 8) Save DE table ----
de_df = sc.get.rank_genes_groups_df(ad_hl, key='p21_wilcoxon', group='high')
de_df_top = de_df.sort_values('pvals_adj', ascending=True).head(500)
de_csv = os.path.join(OUTDIR, "DE_p21_high_vs_low.csv")
de_df_top.to_csv(de_csv, index=False)
print(f"[ok] wrote {de_csv}")

# ===================== A) Volcano (global view) =====================
de_all = sc.get.rank_genes_groups_df(ad_hl, key='p21_wilcoxon', group='high').rename(
    columns={"names":"gene", "logfoldchanges":"logFC", "pvals_adj":"padj", "scores":"score"}
)
de_all["neglog10_padj"] = -np.log10(de_all["padj"].clip(lower=1e-300))
sig   = (de_all["padj"] < 0.05)
big   = (de_all["logFC"].abs() >= 0.25)     # tweak threshold
color = np.where(sig & big, "crimson", np.where(sig, "C0", "lightgrey"))

plt.figure(figsize=(6,5))
plt.scatter(de_all["logFC"], de_all["neglog10_padj"], s=6, c=color, lw=0, alpha=0.8)
plt.axvline( 0.25, color='k', lw=0.5, ls='--'); plt.axvline(-0.25, color='k', lw=0.5, ls='--')
plt.axhline(-np.log10(0.05), color='k', lw=0.5, ls='--')
plt.xlabel("log2 fold-change (high vs low)")
plt.ylabel("-log10(adj p)")
plt.title("p21-high vs p21-low (volcano)")
for g in de_all.loc[sig & big].sort_values("padj").head(10)["gene"]:
    row = de_all.loc[de_all["gene"]==g].iloc[0]
    plt.text(row["logFC"], row["neglog10_padj"], g, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "volcano_p21_high_vs_low.png"), dpi=200)
plt.close()

# ===================== B) Bubble: fraction expressed vs log2FC =====================
# Matrix used for metrics (match your plotting choice)
X = ad_hl.raw.X if (USE_RAW and ad_hl.raw is not None) else (ad_hl.layers[LAYER_PLOT] if not USE_RAW else ad_hl.X)

high_mask = (ad_hl.obs["p21_bin"] == "high").to_numpy()
low_mask  = (ad_hl.obs["p21_bin"] == "low").to_numpy()

# Sparse-safe means & detection fractions (no densify)
if sp.issparse(X):
    X_high = X[high_mask]; X_low = X[low_mask]
    mean_high = np.asarray(X_high.mean(axis=0)).ravel()
    mean_low  = np.asarray(X_low.mean(axis=0)).ravel()
    frac_high = np.asarray((X_high > 0).mean(axis=0)).ravel()
    frac_low  = np.asarray((X_low  > 0).mean(axis=0)).ravel()
else:
    X_high = X[high_mask]; X_low = X[low_mask]
    mean_high = np.asarray(X_high.mean(axis=0)).ravel()
    mean_low  = np.asarray(X_low.mean(axis=0)).ravel()
    frac_high = np.asarray((X_high > 0).mean(axis=0)).ravel()
    frac_low  = np.asarray((X_low  > 0).mean(axis=0)).ravel()

genes = np.array(ad_hl.var_names)
logFC = np.log2((mean_high + 1e-3) / (mean_low + 1e-3))
delta_frac = frac_high - frac_low

df_bub = pd.DataFrame({
    "gene": genes, "logFC": logFC,
    "frac_high": frac_high, "frac_low": frac_low,
    "delta_frac": delta_frac
})

# Prefer full DE table; fall back to de_df_top if needed
if 'de_all' in globals():
    keep_genes = de_all.sort_values("padj").head(300)["gene"].unique()
else:
    keep_genes = de_df_top["names"].unique()
df_bub = df_bub[df_bub["gene"].isin(keep_genes)]

plt.figure(figsize=(6,5))
sizes = (df_bub["frac_high"].clip(0,1) * 400)  # bubble size = fraction expressed in HIGH
scat = plt.scatter(df_bub["logFC"], df_bub["delta_frac"], s=sizes, c=df_bub["logFC"],
                   cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.85, edgecolor='none')
plt.axhline(0, color='k', lw=0.5, ls='--'); plt.axvline(0, color='k', lw=0.5, ls='--')
plt.xlabel("log2FC (high vs low)"); plt.ylabel("Δ fraction expressed (high − low)")
cb = plt.colorbar(scat); cb.set_label("log2FC")
# label a few standouts
for g in df_bub.reindex(df_bub["logFC"].abs().sort_values(ascending=False).head(10).index)["gene"]:
    r = df_bub[df_bub["gene"]==g].iloc[0]
    plt.text(r["logFC"], r["delta_frac"], g, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "bubble_frac_vs_logFC.png"), dpi=200)
plt.close()

# ===================== C) Robust dotplots (non-negative mean log1p) =====================
TOP_N = 20
up_high = de_df_top.head(TOP_N)["names"].tolist()

sc.pl.dotplot(
    ad_hl,
    var_names=up_high,
    groupby="p21_bin",
    use_raw=True,            # ad.raw = log1p-normalized
    standard_scale=None,     # actual mean log1p
    color_map="viridis",     # non-diverging
    vmin=0,                  # no negatives on colorbar
    dot_max=DOT_MAX,
    swap_axes=True,
    save="__p21_high_vs_low_topN_up_in_high.png",
)

# LOW vs HIGH for genes up in LOW
sc.tl.rank_genes_groups(
    ad_hl,
    groupby="p21_bin",
    groups=["low"], reference="high",
    method=RANK_METHOD,
    use_raw=USE_RAW,
    layer=None if USE_RAW else LAYER_PLOT,
    pts=True,
    key_added="p21_wilcoxon_lowvshigh",
)
de_low  = sc.get.rank_genes_groups_df(ad_hl, key="p21_wilcoxon_lowvshigh", group="low")
up_low  = de_low.sort_values("pvals_adj").head(TOP_N)["names"].tolist()

sc.pl.dotplot(
    ad_hl,
    var_names=up_low,
    groupby="p21_bin",
    use_raw=True,
    standard_scale=None,
    color_map="viridis",
    vmin=0,
    dot_max=DOT_MAX,
    swap_axes=True,
    save="__p21_high_vs_low_topN_up_in_low.png",
)

print(f"[info] figures + CSV saved to {OUTDIR}")


XE_BUNDLE = "/projects/chuang-lab/thiesa/SL250282-XR25010_0020938/xeniumranger/region1"
AD_PATH   = "/projects/chuang-lab/thiesa/SL250282-XR25010_0020938/xeniumranger/region1/cell_feature_matrix.h5"        # or Xenium cell_feature_matrix.h5
BINS_PARQ = "/flashscratch/thiesa/align3/p21_krt7_bins.parquet"     # produced by the script

# ==== Imports ====
import os, re, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

# AnnData loader (works for .h5ad and 10x-style .h5)
import scanpy as sc

# ==== Helpers ====
def _idx(x):
    return pd.Index(x.astype(str) if hasattr(x, "astype") else pd.Series(x, dtype=str)).str.strip()

def to_numeric_index(idx_like):
    s = pd.Series(idx_like, dtype=str).str.replace(r"[^\d]+", "", regex=True)
    s = s.where(s.str.len() > 0, np.nan).dropna()
    return pd.Index(s.astype(str))

def strip_affixes_index(idx_like):
    s = pd.Series(idx_like, dtype=str).str.strip()
    s = s.str.replace(r"^(cell[_\-: ]+)", "", regex=True, case=False)
    s = s.str.replace(r"^Cell[_\-: ]+", "", regex=True)
    s = s.str.replace(r"[_\-: ]+cell$", "", regex=True, case=False)
    return pd.Index(s.astype(str))

def load_ad(path):
    if path.endswith(".h5ad"):
        ad = sc.read_h5ad(path)
    elif path.endswith(".h5"):
        ad = sc.read_10x_h5(path)
    else:
        raise ValueError("AD_PATH must be .h5ad or Xenium cell_feature_matrix.h5")
    ad.var_names_make_unique()
    if "feature_types" in ad.var.columns:
        mask = ad.var["feature_types"].astype(str).str.contains("Gene", case=False, na=False)
        ad = ad[:, mask].copy()
    ad.obs_names = _idx(ad.obs_names)
    return ad

def find_cells_parquet(bundle):
    for nm in ["cells.parquet", "cells.parq"]:
        p = Path(bundle) / nm
        if p.exists():
            return str(p)
    return None

# ==== Load inputs ====
print("Loading df_bins …")
df_bins = pd.read_parquet(BINS_PARQ)
mask_idx = _idx(df_bins.index)
print(f"df_bins rows: {len(df_bins)}, sample mask IDs:", list(mask_idx[:5]))

print("\nLoading AnnData …")
ad = load_ad(AD_PATH)
ad_idx = _idx(ad.obs_names)
print(f"AnnData cells: {ad.n_obs}, sample obs_names:", list(ad_idx[:5]))
print("AnnData obs columns (first 12):", list(ad.obs.columns[:12]))

cells_parquet_path = find_cells_parquet(XE_BUNDLE)
cells = None
if cells_parquet_path:
    print(f"\nLoading {cells_parquet_path} …")
    cells = pd.read_parquet(cells_parquet_path)
    have_cols = set(cells.columns)
    print("cells.parquet columns:", sorted(list(have_cols))[:20])
    # Try to show the two most relevant columns if they exist
    if {"cell_id","barcode"}.issubset(have_cols):
        display(cells[["cell_id","barcode"]].head())
    else:
        display(cells.head())
else:
    print("\n[warn] cells.parquet not found in bundle; some mappings may be skipped.")

# ==== Overlap diagnostics ====
def overlap_count(a_idx, b_idx):
    return int(pd.Index(a_idx).intersection(pd.Index(b_idx)).shape[0])

report = []

# 0) direct
n_direct = overlap_count(ad_idx, mask_idx)
report.append(("direct (obs_names vs mask)", n_direct))

# 1) obs['cell_id'] if present
if "cell_id" in ad.obs.columns:
    ad_cell_id = _idx(ad.obs["cell_id"])
    n_cellid = overlap_count(ad_cell_id, mask_idx)
    report.append(("obs['cell_id'] vs mask", n_cellid))
else:
    report.append(("obs['cell_id'] vs mask", -1))

# 2) numeric-only
ad_num   = to_numeric_index(ad_idx)
mask_num = to_numeric_index(mask_idx)
n_numeric = overlap_count(ad_num, mask_num) if (len(ad_num) and len(mask_num)) else 0
report.append(("numeric-only(ad vs mask)", n_numeric))

# 3) prefix/suffix stripping
ad_aff   = strip_affixes_index(ad_idx)
mask_aff = strip_affixes_index(mask_idx)
n_aff = overlap_count(ad_aff, mask_aff)
report.append(("strip-affixes(ad vs mask)", n_aff))

# 4) barcode→cell_id via cells.parquet
n_bar_to_cell = -1
example_map_count = 0
if cells is not None and {"cell_id","barcode"}.issubset(cells.columns):
    b2c = (
        cells[["barcode","cell_id"]]
        .dropna()
        .astype({"barcode": str, "cell_id": str})
        .drop_duplicates(subset=["barcode"])
        .set_index("barcode")["cell_id"]
    )
    # candidate barcodes from AnnData
    cand_bar = ad_idx
    if "barcode" in ad.obs.columns:
        cand_bar = _idx(ad.obs["barcode"])
    mapped = pd.Series(cand_bar).map(b2c).astype("string")
    example_map_count = int(mapped.notna().sum())
    mapped_idx = _idx(mapped[mapped.notna()])
    n_bar_to_cell = overlap_count(mapped_idx, mask_idx)
    report.append(("barcode→cell_id(ad vs mask)", n_bar_to_cell))
else:
    report.append(("barcode→cell_id(ad vs mask)", -1))

# Summarize
rep_df = pd.DataFrame(report, columns=["strategy", "overlap_n"]).set_index("strategy")
print("\n=== Overlap summary ===")
display(rep_df)

print("\nExamples:")
print("  mask_idx[:5]           =", list(mask_idx[:5]))
print("  ad_idx[:5]             =", list(ad_idx[:5]))
if "cell_id" in ad.obs.columns:
    print("  obs['cell_id'][:5]     =", list(_idx(ad.obs['cell_id'])[:5]))
print("  numeric ad[:5]         =", list(ad_num[:5]))
print("  numeric mask[:5]       =", list(mask_num[:5]))
print("  strip-affix ad[:5]     =", list(ad_aff[:5]))
print("  strip-affix mask[:5]   =", list(mask_aff[:5]))
if cells is not None and {"cell_id","barcode"}.issubset(cells.columns):
    print(f"  mapped barcodes → cell_id (non-null count): {example_map_count}")
    print("  first 5 mapped values  =", list(mapped[mapped.notna()].astype(str)[:5]))

# Optional: save a CSV report next to BINS_PARQ
out_csv = str(Path(BINS_PARQ).with_suffix("")) + "_id_overlap_report.csv"
rep_df.to_csv(out_csv)
print(f"\nSaved overlap report to: {out_csv}")
