#!/usr/bin/env python3
"""
Save the top-20 differentially-expressed genes for every Leiden cluster
as a CSV with clusters in rows and rank-ordered gene names in columns.
"""
import scanpy as sc
import pandas as pd
from pathlib import Path

# ────────── paths & basic settings ──────────
H5AD     = "xc_tma_combined_1_with_subclusters.h5ad"   # change if your file lives elsewhere
CL_KEY   = "leiden_1.0"             # the clustering column in .obs
N_TOP    = 100                       # genes to keep per cluster
CSV_OUT  = Path("cluster_top100_DE_genes.csv")

# ────────── load data ──────────
adata = sc.read_h5ad(H5AD)
print(f"{adata.n_obs:,} cells × {adata.n_vars:,} genes loaded")

# ────────── rank genes (wilcoxon) ──────────
rank_key = f"rank_{CL_KEY}"
needs_rerun = (
    rank_key not in adata.uns
    or adata.uns[rank_key]["params"]["n_genes"] < N_TOP
)

if needs_rerun:
    print("Running rank_genes_groups …")
    sc.tl.rank_genes_groups(
        adata,
        groupby   = CL_KEY,
        use_raw   = False,   # work on log1p-normalised X
        layer     = None,
        method    = "wilcoxon",
        n_genes   = N_TOP,
        key_added = rank_key,
    )
else:
    print("Found existing ranking with ≥ 20 genes – re-using it")

# ────────── build cluster × gene table ──────────
rows = []
for cl in adata.obs[CL_KEY].cat.categories:
    top = (
        sc.get.rank_genes_groups_df(
            adata,
            group       = cl,
            key         = rank_key,
            log2fc_min  = None,
            pval_cutoff = None,
        )
        .head(N_TOP)["names"]
        .tolist()
    )
    # pad to exactly N_TOP columns
    rows.append(top + [""] * (N_TOP - len(top)))

columns = [f"Gene_{i+1}" for i in range(N_TOP)]
df = pd.DataFrame(rows,
                  index=adata.obs[CL_KEY].cat.categories,
                  columns=columns)
df.index.name = CL_KEY

# ────────── save ──────────
df.to_csv(CSV_OUT)
print(f"✓ Saved → {CSV_OUT.resolve()}")


# -------------------------------------------------------------------
#  UMAP expression panels:  (Epcam & Abca3)   and   (Nkx2-1)
# -------------------------------------------------------------------
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────
adata_path = "xc_tma_combined_1_with_subclusters.h5ad"     # ← change if needed
figdir     = Path("figures")            # ← same folder as earlier
figdir.mkdir(exist_ok=True)

# ── load AnnData ──────────────────────────────────────────────────
adata = sc.read_h5ad(adata_path)
print(f"✓ {adata.n_obs:,} cells loaded")

# ── ensure requested genes are present ────────────────────────────
genes = ["Epcam", "Abca3", "Egfr"]
missing = [g for g in genes if g not in adata.var_names]
if missing:
    print("⚠️  Warning: gene(s) not found →", ", ".join(missing))

# ── figure 1: Epcam & Abca3 side‑by‑side ──────────────────────────
sc.pl.umap(
    adata,
    color=["Epcam", "Abca3"],
    title=["Epcam expression", "Abca3 expression"],
    frameon=False,
    color_map="viridis",
    vmax="p99",           # clamp outliers for nicer dynamic range
    ncols=2,
    save="_Epcam_Abca3.png",
)

# ── figure 2: Nkx2‑1 alone ────────────────────────────────────────
sc.pl.umap(
    adata,
    color="Egfr",
    title="Egfr expression",
    frameon=False,
    color_map="viridis",
    vmax="p99",
    save="_Egfr.png",
)

plt.show()
print(f"✓ expression panels written to “{figdir.resolve()}”")


# -------------------------------------------------------------------
#  Notebook cell: three UMAP panels (cluster, core, transcript count)
# -------------------------------------------------------------------
import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────
adata_path = "xc_tma_combined_1_with_subclusters.h5ad"     # ← your combined .h5ad file
figdir     = Path("figures")            # ← optional save folder
figdir.mkdir(exist_ok=True)

# ── load & make sure UMAP is present ───────────────────────────────
adata = sc.read_h5ad(adata_path)
print(f"✓ loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")

if "X_umap" not in adata.obsm:
    print("❗ UMAP not found – rebuilding from stored PCA/neighbours")
    sc.tl.umap(adata, random_state=0)

# Tell Scanpy where to save figures when we pass `save=...`
sc.settings.figdir = str(figdir)
sc.settings.set_figure_params(dpi=300)   # high‑res PNGs

# ── panel 1: Leiden clusters ───────────────────────────────────────
sc.pl.umap(
    adata,
    color="leiden_1.0",
    title="Leiden clusters",
    frameon=False,
    legend_loc="on data",
    legend_fontsize=7,
    save="_leiden.png",   # writes figures/umap_leiden.png + .pdf
)

# ── panel 2: core of origin ────────────────────────────────────────
sc.pl.umap(
    adata,
    color="core_ID",
    title="Core of origin",
    frameon=False,
    legend_loc="right margin",
    save="_coreID.png",
)

# ── panel 3: total transcript counts ───────────────────────────────
sc.pl.umap(
    adata,
    color="total_counts",
    title="Total transcript counts",
    frameon=False,
    color_map="viridis",
    vmax="p99",            # cap extreme outliers
    save="_totalCounts.png",
)

plt.show()  # ensures all panels display inline
print(f"✓ plots saved in “{figdir.resolve()}”")






#!/usr/bin/env python3
"""
subcluster6_gene_dotplot.py – Make a dendrogram‑ordered dot‑plot of
genes of interest across the sub‑clusters of parent cluster 6.
"""

import scanpy as sc
import pandas as pd   # (not strictly needed, but you had it)

H5AD_PATH = "xc_tma_combined_1_with_subclusters.h5ad"
adata     = sc.read_h5ad(H5AD_PATH)

# ── parameters ───────────────────────────────────────────────────────────
genes      = ["Cdkn1a", "Cdkn2a", "Cd80", "Cd83",
              "Cd274", "Fscn1", "Ncf1", "Mmp25", "Irf8", "Batf3", "Cd19", "Cd3d", "Cd68"]
cl_key     = "leiden_1_sub"   # ← NEW: sub‑clusters of parent 6
parent_col = "leiden_1.0"     # original clustering column
parent_lab = "1"              # keep only cells from parent 6
layer      = None             # use log‑normalised .X
n_pcs      = 30

# ── 1. subset to the parent‑6 cells ──────────────────────────────────────
mask        = adata.obs[parent_col] == parent_lab
adata_sub   = adata[mask].copy()
print(f"{adata_sub.n_obs:,} cells in parent cluster {parent_lab}")

# ── 2. dendrogram (re‑calculate if needed) ───────────────────────────────
if f"dendrogram_{cl_key}" not in adata_sub.uns:
    sc.tl.dendrogram(adata_sub, groupby=cl_key, n_pcs=n_pcs)

ordered = adata_sub.uns[f"dendrogram_{cl_key}"]["categories_ordered"]
adata_sub.obs[cl_key] = adata_sub.obs[cl_key].cat.reorder_categories(
    ordered, ordered=True
)

# ── 3. dot‑plot on normalised counts (.X) ────────────────────────────────
sc.pl.dotplot(
    adata_sub,
    var_names      = genes,
    groupby        = cl_key,
    layer          = layer,
    standard_scale = "var",
    color_map      = "RdBu_r",
    dot_max        = 0.5,
    dendrogram     = True,
    figsize        = (10, 4 + 0.25 * adata_sub.obs[cl_key].nunique()),
    save           = "subcluster_1_de_norm_dc.png",   # ← NEW output name
    show           = False,
)

print("✓ dot‑plot saved to figures/subcluster_7_de_norm.png")







#!/usr/bin/env python3
"""
cluster1_subs_vs_sub16_neighbourhood.py
───────────────────────────────────────
For every sub‑cluster of parent cluster 1 (`leiden_1_sub`):

    • count how many of its cells are within a fixed *radius*
      of ANY cell in sub‑cluster 16 (same coordinate units).
    • report  n_within_radius  and  fraction_within_radius.

Change RADIUS if you prefer a different distance cut‑off.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# ─── adjustable parameters ─────────────────────────────────────────────
H5AD_PATH  = "xc_tma_combined_1_with_subclusters.h5ad"
ORIG_CL    = "leiden_1.0"       # parent Leiden column
SUB_CL     = "leiden_1_sub"     # sub‑clusters inside parent 1
TARGET_SUB = "16"               # sub‑cluster we measure distances to
SPATIAL    = "spatial"          # obsm key with coords
PARENT_1   = "1"
RADIUS     = 5.0                # <-- physical distance cut‑off

# ─── load & subset parent‑1 cells ──────────────────────────────────────
adata = sc.read_h5ad(H5AD_PATH)
mask_parent1 = adata.obs[ORIG_CL] == PARENT_1
adata1       = adata[mask_parent1]

# ─── identify target sub‑cluster 16 and build KD‑tree ──────────────────
mask_target  = adata1.obs[SUB_CL] == TARGET_SUB
if mask_target.sum() == 0:
    raise ValueError(
        f"No cells labelled '{TARGET_SUB}' found in '{SUB_CL}'. "
        "Check the label (e.g. maybe '16.0')."
    )

coords_target = adata1.obsm[SPATIAL][mask_target]
tree_target   = cKDTree(coords_target)

print(f"{adata1.n_obs:,} cells in parent 1; "
      f"{mask_target.sum():,} of them are in sub‑cluster {TARGET_SUB}")

# ─── for every other sub‑cluster: neighbourhood counts ────────────────
results = []
for sub in adata1.obs[SUB_CL].cat.categories:
    mask_sub = adata1.obs[SUB_CL] == sub
    coords   = adata1.obsm[SPATIAL][mask_sub]

    # list of lists: indices of target‑cells within RADIUS for each query point
    neighbours = tree_target.query_ball_point(coords, r=RADIUS)
    hits       = np.fromiter((len(n) > 0 for n in neighbours), bool)

    n_total    = hits.size
    n_prox     = hits.sum()
    frac_prox  = n_prox / n_total if n_total > 0 else np.nan

    results.append(
        dict(
            subcluster            = sub,
            n_cells               = n_total,
            n_within_radius       = n_prox,
            frac_within_radius    = frac_prox,
        )
    )

df = (pd.DataFrame(results)
        .sort_values("frac_within_radius", ascending=False)
        .reset_index(drop=True))

# ─── save & display ───────────────────────────────────────────────────
out_dir = Path("tables"); out_dir.mkdir(exist_ok=True)
csv_out = out_dir / "cluster1_subs_vs_sub16_neighbourhood.csv"
df.to_csv(csv_out, index=False)

print(f"\nNeighbourhood (radius {RADIUS}) summary:")
print(df)
print(f"\n✓ table written to {csv_out.resolve()}")








import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── edit these two lines if your file/location differs ─────────────────
CSV_PATH   = Path("tables/cluster1_subs_vs_sub16_neighbourhood.csv")
TARGET_SUB = "16"          # use "16.0" if that’s the exact label

# ── load results table ─────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# robust handling if the label was stored as "16.0"
if TARGET_SUB not in df["subcluster"].astype(str).values and "16.0" in df["subcluster"].astype(str).values:
    TARGET_SUB = "16.0"

# ── drop the target sub‑cluster itself ─────────────────────────────────
plot_df = df[df["subcluster"].astype(str) != TARGET_SUB].copy()

# sort by highest proximity fraction for nicer plotting
plot_df.sort_values("frac_within_radius", ascending=False, inplace=True)

# ── bar chart ──────────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.bar(plot_df["subcluster"].astype(str), plot_df["frac_within_radius"])
plt.xlabel("Sub‑cluster (leiden_1_sub)")
plt.ylabel("Fraction of cells within radius of sub‑cluster 16")
plt.title("Proximity to sub‑cluster 16 (target removed)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# save or show – your choice
plt.savefig("sub1_subs_vs_sub16_neighbourhood_bar.png", dpi=200)
plt.show()






#!/usr/bin/env python3
"""
sub1_subs_vs_sub16_distance.py
──────────────────────────────
Ranks every sub‑cluster of parent cluster 1 by how close it lies to
sub‑cluster 16 (all in physical coordinates stored in obsm["spatial"]).

Output: tables/cluster1_subs_vs_sub16_distances.csv
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# ───‑ parameters you might tweak ────────────────────────────────────────
H5AD_PATH  = "xc_tma_combined_1_with_subclusters.h5ad"
PARENT_COL = "leiden_1.0"     # original clustering column
SUB_COL    = "leiden_1_sub"   # sub‑clusters of parent 1
TARGET_SUB = "16"             # the sub‑cluster we measure distances to
SPATIAL    = "spatial"        # obsm key with x,y (or x,y,z) coords
PARENT_1   = "1"              # label of parent cluster 1

# ───‑ load object & sanity checks ───────────────────────────────────────
adata = sc.read_h5ad(H5AD_PATH)
for key, where in [(SPATIAL, "obsm"), (SUB_COL, "obs"), (PARENT_COL, "obs")]:
    assert (key in (adata.obsm if where == "obsm" else adata.obs)), \
        f"❌ '{key}' not found in adata.{where}"

# ───‑ keep only parent‑1 cells ─────────────────────────────────────────
mask_parent1 = adata.obs[PARENT_COL] == PARENT_1
adata1       = adata[mask_parent1]

print(f"{adata1.n_obs:,} cells in parent cluster {PARENT_1}")

# ───‑ identify the target sub‑cluster 16 ───────────────────────────────
mask_target = adata1.obs[SUB_COL] == TARGET_SUB
if mask_target.sum() == 0:
    raise ValueError(
        f"No cells labelled '{TARGET_SUB}' found in '{SUB_COL}'. "
        "Verify the label (e.g. might be '16.0')."
    )

coords_target = adata1.obsm[SPATIAL][mask_target]
tree_target   = cKDTree(coords_target)

# ───‑ measure distances for every other sub‑cluster ────────────────────
results = []
for sub_label in adata1.obs[SUB_COL].cat.categories:
    if sub_label == TARGET_SUB:
        continue                                 # skip self‑comparison
    mask_sub = adata1.obs[SUB_COL] == sub_label
    coords   = adata1.obsm[SPATIAL][mask_sub]
    if coords.size == 0:                         # empty sub‑cluster guard
        continue
    dists, _ = tree_target.query(coords, k=1)    # nearest‑neighbour distances
    results.append(
        dict(
            subcluster   = sub_label,
            n_cells      = coords.shape[0],
            min_dist     = dists.min(),
            median_dist  = np.median(dists),
            mean_dist    = dists.mean(),
        )
    )

# ───‑ tidy, rank by median distance, save CSV ──────────────────────────
df = (pd.DataFrame(results)
        .sort_values("median_dist")
        .reset_index(drop=True))

out_dir = Path("tables"); out_dir.mkdir(exist_ok=True)
csv_out = out_dir / "cluster1_subs_vs_sub16_distances.csv"
df.to_csv(csv_out, index=False)

print("\nDistance to sub‑cluster 16 (same units as obsm['spatial']):")
print(df)
print(f"\n✓ table written to {csv_out.resolve()}")






import pandas as pd, scanpy as sc
adata = sc.read_h5ad("xc_tma_combined_1_with_subclusters.h5ad")
print("Available clusters in leiden_1.0:")
print(adata.obs["leiden_1.0"].cat.categories.tolist())






import numpy as np, pandas as pd
from scipy.spatial.distance import cdist

X_pca = adata.obsm["X_pca"]                # 40-PC space from the original run
mask_cl0  = adata.obs["leiden_1.0"] == "0"
mask_sub  = adata.obs["leiden_1.0"] == "1"

centroid_0 = X_pca[mask_cl0].mean(axis=0, keepdims=True)   # (1 × 40)

sub_labels = adata.obs.loc[mask_sub, "leiden_1_sub"]       # now present
centroids_sub = (
    pd.DataFrame(X_pca[mask_sub], index=sub_labels)
      .groupby(level=0)
      .mean()
      .values                                              # (k × 40)
)

dists = cdist(centroid_0, centroids_sub)[0]
closest = sub_labels.cat.categories[dists.argmin()]

print("\nDistances from cluster 0 centroid (40-PC space):")
for lbl, d in sorted(zip(sub_labels.cat.categories, dists), key=lambda x: x[1]):
    print(f"  sub-{lbl:>2}: {d:0.3f}")

print(f"\n⚡  Sub-cluster {closest} is transcriptionally closest to cluster 0.")







import scanpy as sc
import pandas as pd

H5AD_PATH   = "xc_tma_combined_1_with_subclusters.h5ad"   # ← path to combined AnnData
adata = sc.read_h5ad(H5AD_PATH)

# ── parameters ───────────────────────────────────────────────────────────
genes   = ["Chil1", "Lamp3", "Dusp6", "Lgi3", "Arg1", "Tpi1", "Kras", "Egfr", "Epcam", "Abca3",
           "Chil3", "Cdkn1a", "Cdkn2a", "Lamp3", "Cd36", "Xist", "Aqp5", "Mdm2", "Trp53", "Col18a1", "Ccna2", "Mki67"]
cl_key  = "leiden_1.0"        # cluster column
layer   = "counts"            # raw counts → nicer dynamic range
n_pcs   = 40                  # #PCs for clustering the clusters themselves

# ── 1. dendrogram (only once) ────────────────────────────────────────────
if f"dendrogram_{cl_key}" not in adata.uns:
    sc.tl.dendrogram(adata, groupby=cl_key, n_pcs=n_pcs)

# ── 2. reorder the categorical according to that dendrogram ─────────────
ordered = adata.uns[f"dendrogram_{cl_key}"]["categories_ordered"]
adata.obs[cl_key] = adata.obs[cl_key].cat.reorder_categories(
    ordered, ordered=True
)

# (optional) confirm the new order
print("Cluster order:", list(adata.obs[cl_key].cat.categories))

# ── 3. dot-plot with dendrogram ─────────────────────────────────────────
sc.pl.dotplot(
    adata,
    var_names      = genes,
    groupby        = cl_key,
    standard_scale = "var",           # z-score across clusters
    color_map      = "RdBu_r",
    dot_max        = 0.5,
    dendrogram     = True,            # << draws tree & keeps order
    figsize        = (10, 4 + 0.25*adata.obs[cl_key].nunique()),
    save           = "cell_cycle_plot_1_new.png",
    show           = False
)



# %% [markdown]
# ## 5  Stacked bar‑chart – sub‑cluster composition per core
# %%
import pandas as pd

# ── 1. tabulate counts ───────────────────────────────────────────────────
# rows = cores, columns = sub‑clusters
counts = (
    pd.crosstab(
        obs_plot[CORE_KEY],          # ← ONLY the plottable cells
        obs_plot[SUB_KEY]
    )
    .reindex(columns=subcats, fill_value=0)        # canonical sub‑cluster order
    .sort_index()                                  # same core order as before
)

# convert to fractions (so each bar sums to 1); remove `fractions = …`
fractions = counts.div(counts.sum(axis=1), axis=0)

# ── 2. stacked bar‑plot ──────────────────────────────────────────────────
fig_w = 0.7 * len(fractions) if len(fractions) > 4 else 4
fig, ax = plt.subplots(figsize=(fig_w, 4))

bottom = np.zeros(len(fractions))     # running baseline for stacking
x_pos  = np.arange(len(fractions))    # one bar per core

for cl in subcats:
    heights = fractions[cl].values
    color   = colors[cat_to_code[cl]]     # identical palette as scatter plot

    ax.bar(
        x_pos,
        heights,
        bottom=bottom,
        color=color,
        width=0.8,
        edgecolor="none",
        label=cl
    )
    bottom += heights                    # update baseline

# ── 3. cosmetics ─────────────────────────────────────────────────────────
ax.set_xticks(x_pos)
ax.set_xticklabels(fractions.index, rotation=45, ha="right")
ax.set_ylabel("Fraction of cells")
ax.set_xlabel("Core ID")
ax.set_ylim(0, 1)
ax.set_title("Sub‑cluster representation within parent 1 per core")

# legend outside the plot for readability
ax.legend(title="leiden_1_sub", bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout()
fig.savefig("cluster1_sub_fractions_per_core.png", dpi=150)
plt.show()




#!/usr/bin/env python3
"""
Save the top‑N differentially‑expressed genes (Wilcoxon) for every
sub‑cluster of parent cluster 1 as a CSV.

Rows  = sub‑clusters (e.g. '1_a', '1_b', …)
Cols  = Gene_1 … Gene_N (rank‑ordered by significance)
"""
import scanpy as sc
import pandas as pd
from pathlib import Path

# ────────── paths & settings ──────────
H5AD        = "xc_tma_combined_1_with_subclusters.h5ad"
SUB_KEY     = "leiden_1_sub"      # sub‑clusters of parent 1
EXCLUDE_CAT = "outside_1"         # cells not in parent 1
N_TOP       = 25                 # genes per sub‑cluster
CSV_OUT     = Path("cluster1_subclusters_top20_genes.csv")

# ────────── load & subset ──────────
adata = sc.read_h5ad(H5AD)
print(f"{adata.n_obs:,} cells × {adata.n_vars:,} genes loaded")

keep_mask = adata.obs[SUB_KEY] != EXCLUDE_CAT
adata_sub = adata[keep_mask].copy()
adata_sub.obs[SUB_KEY] = (
    adata_sub.obs[SUB_KEY].astype("category")
            .cat.remove_unused_categories()
)
print(f"Analysing {adata_sub.n_obs:,} cells in "
      f"{len(adata_sub.obs[SUB_KEY].cat.categories)} sub‑clusters")

# ────────── rank genes (Wilcoxon) ──────────
rank_key = f"rank_{SUB_KEY}_{N_TOP}"
sc.tl.rank_genes_groups(
    adata_sub,
    groupby   = SUB_KEY,
    use_raw   = False,
    layer     = None,
    method    = "wilcoxon",
    n_genes   = N_TOP,
    key_added = rank_key,
)
print("Finished ranking genes")

# ────────── build sub‑cluster × gene table ──────────
rows = []
for sub in adata_sub.obs[SUB_KEY].cat.categories:
    top = (
        sc.get.rank_genes_groups_df(
            adata_sub,
            group       = sub,
            key         = rank_key,
            log2fc_min  = None,
            pval_cutoff = None,
        )
        .head(N_TOP)["names"]
        .tolist()
    )
    rows.append(top + [""] * (N_TOP - len(top)))

columns = [f"Gene_{i+1}" for i in range(N_TOP)]
df = pd.DataFrame(rows,
                  index=adata_sub.obs[SUB_KEY].cat.categories,
                  columns=columns)
df.index.name = SUB_KEY

# ────────── save ──────────
df.to_csv(CSV_OUT)
print(f"✓ Saved → {CSV_OUT.resolve()}")





# %% [markdown]
# ## Sub-cluster parent Leiden 1 on log-normalised counts
#    – uses .X for clustering, keeps raw counts in .layers["counts"]
# %%
import scanpy as sc
import numpy as np

COMBINED_H5AD = "xc_tma_combined_1_with_subclusters.h5ad"   # ← produced by xc_tma_cluster.py
PARENT_CL_KEY = "leiden_1.0"             # column with original clusters
PARENT_LABEL  = "1"                      # the parent cluster to zoom into
SUB_KEY       = "leiden_1_sub"           # new column for sub-clusters
N_PCS         = 30
RESOLUTION    = 0.5                      # tune for #sub-clusters
N_TOP_G       = 5                        # genes per sub-cluster in dot-plot

# ────────────────────────── 1. load & subset ────────────────────────────
adata = sc.read_h5ad(COMBINED_H5AD)

mask        = adata.obs[PARENT_CL_KEY] == PARENT_LABEL
adata_sub   = adata[mask].copy()               # .X is already log-norm’d
print(f"Subsetting cluster {PARENT_LABEL}: {adata_sub.n_obs} cells")

# ────────────────────────── 2. recluster on log-data ────────────────────
sc.pp.highly_variable_genes(
    adata_sub, n_top_genes=3000, flavor="seurat_v3"
)
sc.pp.scale(adata_sub, zero_center=False)       # re-scale log counts
sc.tl.pca(adata_sub, n_comps=N_PCS, svd_solver="arpack")
sc.pp.neighbors(adata_sub, n_pcs=N_PCS, random_state=0)
sc.tl.leiden(adata_sub, resolution=RESOLUTION,
             key_added=SUB_KEY, random_state=0)
sc.tl.umap(adata_sub, random_state=0)

print("Sub-clusters:", adata_sub.obs[SUB_KEY].cat.categories.tolist())

# ────────────────────────── 3. DE (raw counts) ──────────────────────────
# keep raw UMI counts for Wilcoxon statistics
sc.tl.rank_genes_groups(
    adata_sub,
    groupby   = SUB_KEY,
    layer     = "counts",     # raw counts saved in the main script
    n_genes   = N_TOP_G,
    method    = "wilcoxon",
    key_added = "rank_genes_"+SUB_KEY,
)

# build list-of-lists: 5 genes per sub-cluster
rg             = adata_sub.uns["rank_genes_"+SUB_KEY]
sub_labels     = rg["names"].dtype.names
genes_by_sub   = [
    [g for g in rg["names"][lab][:N_TOP_G] if g in adata_sub.var_names]
    for lab in sub_labels
]

# ────────────────────────── 4. hierarchical dot-plot ────────────────────
#   (reorder rows by dendrogram, group columns by “home” cluster)
sc.tl.dendrogram(adata_sub, groupby=SUB_KEY, n_pcs=N_PCS)
ordered_subs = adata_sub.uns[f"dendrogram_{SUB_KEY}"]["categories_ordered"]
adata_sub.obs[SUB_KEY] = adata_sub.obs[SUB_KEY].cat.reorder_categories(
    ordered_subs, ordered=True
)
genes_by_sub = [genes_by_sub[sub_labels.index(lbl)] for lbl in ordered_subs]

# flatten gene list & block positions
flat_genes, var_pos = [], []
pos = 0
for block in genes_by_sub:
    flat_genes.extend(block)
    var_pos.append((pos, pos + len(block) - 1))
    pos += len(block)

sc.pl.dotplot(
    adata_sub,
    var_names          = flat_genes,
    groupby            = SUB_KEY,
    layer              = "counts",        # colour & dot-size from raw counts
    standard_scale     = "var",
    color_map          = "RdBu_r",
    dot_max            = 0.4,
    dendrogram         = True,
    var_group_positions= var_pos,
    var_group_labels   = ordered_subs,
    figsize            = (0.8*len(flat_genes)+6,
                           2+0.35*adata_sub.obs[SUB_KEY].nunique()),
    save               = "_cluster1_subclusters_hier_dotplot.png",
    show               = False,
)

# After sub-clustering is finished
adata.obs.loc[mask, "leiden_1_sub"] = adata_sub.obs["leiden_1_sub"].values
adata.write_h5ad("xc_tma_combined_1_with_subclusters.h5ad")

print("✓ dot-plot saved to figures/_cluster1_subclusters_hier_dotplot.png")






import scanpy as sc
import pandas as pd

H5AD_PATH = "xc_tma_combined_1_with_subclusters.h5ad"
adata     = sc.read_h5ad(H5AD_PATH)

# ── parameters ───────────────────────────────────────────────────────────
genes   = ["Cdkn1a", "Cdkn2a", "Pecam1", "Vwf", "Cdh5", "Ccl2", "Cxcl2"]
cl_key  = "leiden_1_sub"
layer   = None            # <<< use normalised counts in .X
n_pcs   = 30

# ── 1. dendrogram (only once) ────────────────────────────────────────────
if f"dendrogram_{cl_key}" not in adata.uns:
    sc.tl.dendrogram(adata, groupby=cl_key, n_pcs=n_pcs)

# ── 2. reorder the categorical according to that dendrogram ─────────────
ordered = adata.uns[f"dendrogram_{cl_key}"]["categories_ordered"]
adata.obs[cl_key] = adata.obs[cl_key].cat.reorder_categories(
    ordered, ordered=True
)

# ── (optional) differential expression on *normalised* data ─────────────
# If you also want a DE table (top genes per sub-cluster) on the log-norm values:
# sc.tl.rank_genes_groups(
#     adata,
#     groupby   = cl_key,
#     layer     = layer,        # None  -> use .X
#     use_raw   = False,
#     n_genes   = 100,          # or whatever you need
#     method    = "wilcoxon",
#     key_added = "rank_genes_norm_"+cl_key,
# )

# ── 3. dot-plot with dendrogram (normalised layer) ──────────────────────
sc.pl.dotplot(
    adata,
    var_names      = genes,
    groupby        = cl_key,
    layer          = layer,          # None  -> .X (log-norm)
    standard_scale = "var",
    color_map      = "RdBu_r",
    dot_max        = 0.5,
    dendrogram     = True,
    figsize        = (10, 4 + 0.25 * adata.obs[cl_key].nunique()),
    save           = "subcluster_1_de_norm.png",
    show           = False,
)
print("✓ dot-plot saved to figures/subcluster_1_de_norm.png")


