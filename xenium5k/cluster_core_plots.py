import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ===================== SETTINGS =====================
h5ad_path   = "/flashscratch/thiesa/codex2/xc_tma_combined_include_control_03.h5ad"
outdir      = Path("cluster_dotplots")
cluster_key = "leiden_1.0"          # change if needed

n_de_genes  = 12                    # per cluster
n_top_expr  = 12                    # per cluster

method      = "wilcoxon"            # good default
key_added   = f"rank_{cluster_key}" # results key in .uns

figsize_de  = (6.0, 3.0)            # keep plots not too large
figsize_top = (6.0, 3.0)
dpi         = 200

# Optional: filter out noisy housekeeping for the "top expressed" list
exclude_prefixes = ("mt-", "MT-")   # mitochondrial
exclude_exact = set()              # e.g. {"MALAT1"} if you want
# ====================================================

outdir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(h5ad_path)

if cluster_key not in adata.obs.columns:
    raise ValueError(f"'{cluster_key}' not found in adata.obs. Available: {list(adata.obs.columns)[:20]} ...")

# Ensure categorical & stable order
adata.obs[cluster_key] = adata.obs[cluster_key].astype(str).astype("category")
clusters = adata.obs[cluster_key].cat.categories.tolist()

# We want dotplots/DE to use log-normalized unscaled values
if adata.raw is None:
    raise ValueError(
        "adata.raw is None. Your 'better' script usually sets adata.raw after log1p.\n"
        "If you truly don't have .raw, tell me and I'll give you a safe reconstruction from layers['counts']."
    )

# -------------------- 1) Differential expression --------------------
# Compute once for all clusters vs rest
sc.tl.rank_genes_groups(
    adata,
    groupby=cluster_key,
    method=method,
    use_raw=True,
    key_added=key_added,
)

# Save DE tables (one big CSV)
dfs = []
for g in clusters:
    df = sc.get.rank_genes_groups_df(adata, group=g, key=key_added)
    df.insert(0, "cluster", g)
    dfs.append(df)
de_all = pd.concat(dfs, ignore_index=True)
de_csv = outdir / f"DE_{cluster_key}_{method}.csv"
de_all.to_csv(de_csv, index=False)
print("Wrote:", de_csv)

# One DE dotplot per cluster (genes = that cluster's top DE)
for g in clusters:
    # rank_genes_groups_dotplot is built for this
    dp = sc.pl.rank_genes_groups_dotplot(
        adata,
        groupby=cluster_key,
        key=key_added,
        groups=[g],
        n_genes=n_de_genes,
        use_raw=True,
        figsize=figsize_de,
        show=False,
    )
    # Save figure
    plt.gcf().savefig(outdir / f"dotplot_DE_{cluster_key}_cluster_{g}.png",
                      dpi=dpi, bbox_inches="tight")
    plt.close()

print("Saved DE dotplots.")

# -------------------- 2) Top expressed genes per cluster --------------------
# Use mean expression within each cluster from adata.raw
raw = adata.raw  # AnnData "raw" view
X = raw.X
var_names = np.array(raw.var_names)

# Helpful: precompute mask for excluded genes
exclude_mask = np.zeros(len(var_names), dtype=bool)
for p in exclude_prefixes:
    exclude_mask |= np.char.startswith(var_names.astype(str), p)
if exclude_exact:
    exclude_mask |= np.isin(var_names, list(exclude_exact))

# Convert to CSR if sparse for faster slicing/means
try:
    import scipy.sparse as sp
    if sp.issparse(X) and not sp.isspmatrix_csr(X):
        X = X.tocsr()
except Exception:
    pass

# Compute top expressed and plot dotplot per cluster
for g in clusters:
    idx = (adata.obs[cluster_key].astype(str).values == str(g))
    if idx.sum() == 0:
        continue

    Xm = X[idx]
    # mean per gene
    mu = np.asarray(Xm.mean(axis=0)).ravel()

    # apply exclusion and pick top genes
    mu2 = mu.copy()
    mu2[exclude_mask] = -np.inf

    top_idx = np.argpartition(-mu2, range(min(n_top_expr, len(mu2))))[:n_top_expr]
    top_idx = top_idx[np.argsort(-mu2[top_idx])]
    top_genes = var_names[top_idx].tolist()

    dp = sc.pl.dotplot(
        adata,
        var_names=top_genes,
        groupby=cluster_key,
        use_raw=True,
        categories_order=clusters,   # consistent cluster order across all images
        figsize=figsize_top,
        show=False,
    )
    plt.gcf().savefig(outdir / f"dotplot_TOP_EXPR_{cluster_key}_cluster_{g}.png",
                      dpi=dpi, bbox_inches="tight")
    plt.close()

print("Saved top-expressed dotplots.")
print("All outputs in:", outdir.resolve())
