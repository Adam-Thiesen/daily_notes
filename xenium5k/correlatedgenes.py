%matplotlib inline
import os
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

H5AD_PATH   = "xc_5k_all_runs.h5ad" 
adata = sc.read_h5ad(H5AD_PATH)

# --- Spatial cluster plots per experiment (only selected clusters; no background image) ---
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ================== PARAMS ==================
CLUSTER_KEY    = "leiden_0.25"   # e.g. "leiden_0.2" or your chosen column
OUTDIR         = "spatial_clusters_selected2"
FIGSIZE        = (8, 8)          # per-experiment figure size (inches)
ALPHA          = 0.85            # point alpha
FLIP_Y         = True            # flip y-axis (image-style coordinates)
JITTER         = 0.0             # small random jitter in coordinate units (e.g., 0.5)
POINT_SIZE     = None            # None = auto by density; else a number (e.g., 1.0, 2.0, 3.0)
BG_COLOR       = "white"         # figure background
EDGE_COLOR     = "none"          # outline around points; try 'black' for thin edges
DPI            = 200             # output resolution
EXPERIMENTS    = None            # None = all; or list like ["section1","section2"]

# Which clusters to plot (others will be hidden entirely)
TARGET_CLUSTERS = [6, 10, 12, 13, 16, 19, 4, 8, 9, 11, 21]  # updated to include additional clusters  # <- customize as needed

# Optional custom palette just for these target clusters (same order as TARGET_CLUSTERS)
# Provide either a list of hex colors, or set to None to auto-generate a palette
CUSTOM_PALETTE = None  # auto-generate distinct colors for all selected clusters
# ============================================

# ---- 0) Basic checks ----
if CLUSTER_KEY not in adata.obs.columns:
    raise KeyError(f"{CLUSTER_KEY!r} not found in adata.obs")

if "experiment_id" not in adata.obs.columns:
    raise KeyError("'experiment_id' not found in adata.obs (set in the QC/cluster script)")

# Coordinate source: prefer obsm['spatial'], else obs['x','y']
if "spatial" in adata.obsm_keys():
    XY = adata.obsm["spatial"]
    xy_names = ("x","y")
else:
    if not {"x","y"}.issubset(adata.obs.columns):
        raise KeyError("No coordinates found: need obsm['spatial'] or obs['x','y'].")
    XY = adata.obs[["x","y"]].to_numpy()
    xy_names = ("x","y")

# ---- 1) Ensure categorical cluster with stable order ----
cl = adata.obs[CLUSTER_KEY].astype("category")
adata.obs[CLUSTER_KEY] = cl.cat.remove_unused_categories()

# normalize cluster labels to strings for matching
all_clusters = list(adata.obs[CLUSTER_KEY].cat.categories)
all_clusters_str = [str(c) for c in all_clusters]

# convert TARGET_CLUSTERS to strings (so it works whether categories are int-like or str)
target_str = [str(c) for c in TARGET_CLUSTERS]

# keep only target clusters that actually exist in the data (and preserve requested order)
selected_clusters = [c for c in target_str if c in all_clusters_str]
if not selected_clusters:
    raise ValueError(
        f"None of the requested TARGET_CLUSTERS {TARGET_CLUSTERS} were found in {CLUSTER_KEY}. "
        f"Available: {all_clusters}"
    )

# ---- 2) Build a palette JUST for the selected clusters ----
def categorical_palette(n):
    # Use tab10 for up to 10 distinct, otherwise fall back to hsv
    if n <= 10:
        cmap = mpl.cm.get_cmap("tab10", n)
        return [mpl.colors.to_hex(cmap(i)) for i in range(n)]
    else:
        cmap = mpl.cm.get_cmap("hsv", n)
        return [mpl.colors.to_hex(cmap(i)) for i in range(n)]

if CUSTOM_PALETTE is not None:
    if len(CUSTOM_PALETTE) < len(selected_clusters):
        raise ValueError(
            f"CUSTOM_PALETTE has {len(CUSTOM_PALETTE)} colors but {len(selected_clusters)} clusters were requested."
        )
    palette = CUSTOM_PALETTE[: len(selected_clusters)]
else:
    palette = categorical_palette(len(selected_clusters))

cluster_to_color = dict(zip(selected_clusters, palette))

# ---- 3) Auto point size by density (unless user-specified) ----
def auto_point_size(n_cells):
    if n_cells <= 50_000:
        return 2.0
    if n_cells <= 200_000:
        return 1.2
    if n_cells <= 1_000_000:
        return 0.6
    return 0.4

# ---- 4) Make output dir ----
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# ---- 5) Optionally filter experiments ----
exp_cats = adata.obs["experiment_id"].astype(str)
if EXPERIMENTS is None:
    experiments = sorted(exp_cats.unique())
else:
    experiments = [e for e in EXPERIMENTS if e in set(exp_cats.unique())]
    missing = [e for e in EXPERIMENTS if e not in set(exp_cats.unique())]
    if missing:
        print(f"[warn] Experiments not found and skipped: {missing}")

# ---- 6) Plot per experiment ----
for exp in experiments:
    mask_exp = (exp_cats == exp).values
    n_exp = int(mask_exp.sum())
    if n_exp == 0:
        continue

    xy = XY[mask_exp].astype(float, copy=True)
    if JITTER and JITTER > 0:
        xy += np.random.default_rng(0).normal(scale=JITTER, size=xy.shape)

    # Determine point size
    s = POINT_SIZE if POINT_SIZE is not None else auto_point_size(n_exp)

    # Prepare figure
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Only plot the selected clusters
    cls_exp = adata.obs.loc[mask_exp, CLUSTER_KEY].astype(str).to_numpy()

    handles = []
    labels = []
    for c in selected_clusters:
        cmask = (cls_exp == c)
        if not cmask.any():
            continue
        ax.scatter(
            xy[cmask, 0], xy[cmask, 1],
            s=s, c=cluster_to_color[c], alpha=ALPHA,
            edgecolors=EDGE_COLOR, linewidths=0.0, rasterized=True
        )
        handles.append(
            mpl.lines.Line2D([0],[0], marker="o", color="none",
                              markerfacecolor=cluster_to_color[c],
                              markersize=6, label=c)
        )
        labels.append(c)

    # Axes style
    ax.set_title(
        f"{exp}  •  n={n_exp:,} cells  •  {CLUSTER_KEY} (showing {len(selected_clusters)} selected)",
        fontsize=12
    )
    ax.set_xlabel(xy_names[0]); ax.set_ylabel(xy_names[1])

    # Flip Y if desired (image coordinate style)
    if FLIP_Y:
        ax.invert_yaxis()

    # Equal aspect so spatial geometry isn't distorted
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # Tidy ticks
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend outside
    if handles:
        ax.legend(
            handles=handles, labels=labels, title=f"{CLUSTER_KEY} (selected)",
            loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, ncol=1
        )

    # Tight layout & save
    fname = Path(OUTDIR) / f"spatial_clusters_selected__{exp}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {fname}")

print("Done.")


# --- Spatial cluster plots per experiment (only selected clusters; no background image) ---
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ================== PARAMS ==================
CLUSTER_KEY    = "leiden_0.25"   # e.g. "leiden_0.2" or your chosen column
OUTDIR         = "spatial_clusters_selected3"
FIGSIZE        = (8, 8)          # per-experiment figure size (inches)
ALPHA          = 0.85            # point alpha
FLIP_Y         = True            # flip y-axis (image-style coordinates)
JITTER         = 0.0             # small random jitter in coordinate units (e.g., 0.5)
POINT_SIZE     = None            # None = auto by density; else a number (e.g., 1.0, 2.0, 3.0)
BG_COLOR       = "white"         # figure background
EDGE_COLOR     = "none"          # outline around points; try 'black' for thin edges
DPI            = 200             # output resolution
EXPERIMENTS    = None            # None = all; or list like ["section1","section2"]

# Which clusters to plot (others will be hidden entirely)
TARGET_CLUSTERS = [0, 1, 2, 3, 5, 7, 20]  # updated to include additional clusters  # <- customize as needed

# Optional custom palette just for these target clusters (same order as TARGET_CLUSTERS)
# Provide either a list of hex colors, or set to None to auto-generate a palette
CUSTOM_PALETTE = None  # auto-generate distinct colors for all selected clusters
# ============================================

# ---- 0) Basic checks ----
if CLUSTER_KEY not in adata.obs.columns:
    raise KeyError(f"{CLUSTER_KEY!r} not found in adata.obs")

if "experiment_id" not in adata.obs.columns:
    raise KeyError("'experiment_id' not found in adata.obs (set in the QC/cluster script)")

# Coordinate source: prefer obsm['spatial'], else obs['x','y']
if "spatial" in adata.obsm_keys():
    XY = adata.obsm["spatial"]
    xy_names = ("x","y")
else:
    if not {"x","y"}.issubset(adata.obs.columns):
        raise KeyError("No coordinates found: need obsm['spatial'] or obs['x','y'].")
    XY = adata.obs[["x","y"]].to_numpy()
    xy_names = ("x","y")

# ---- 1) Ensure categorical cluster with stable order ----
cl = adata.obs[CLUSTER_KEY].astype("category")
adata.obs[CLUSTER_KEY] = cl.cat.remove_unused_categories()

# normalize cluster labels to strings for matching
all_clusters = list(adata.obs[CLUSTER_KEY].cat.categories)
all_clusters_str = [str(c) for c in all_clusters]

# convert TARGET_CLUSTERS to strings (so it works whether categories are int-like or str)
target_str = [str(c) for c in TARGET_CLUSTERS]

# keep only target clusters that actually exist in the data (and preserve requested order)
selected_clusters = [c for c in target_str if c in all_clusters_str]
if not selected_clusters:
    raise ValueError(
        f"None of the requested TARGET_CLUSTERS {TARGET_CLUSTERS} were found in {CLUSTER_KEY}. "
        f"Available: {all_clusters}"
    )

# ---- 2) Build a palette JUST for the selected clusters ----
def categorical_palette(n):
    # Use tab10 for up to 10 distinct, otherwise fall back to hsv
    if n <= 10:
        cmap = mpl.cm.get_cmap("tab10", n)
        return [mpl.colors.to_hex(cmap(i)) for i in range(n)]
    else:
        cmap = mpl.cm.get_cmap("hsv", n)
        return [mpl.colors.to_hex(cmap(i)) for i in range(n)]

if CUSTOM_PALETTE is not None:
    if len(CUSTOM_PALETTE) < len(selected_clusters):
        raise ValueError(
            f"CUSTOM_PALETTE has {len(CUSTOM_PALETTE)} colors but {len(selected_clusters)} clusters were requested."
        )
    palette = CUSTOM_PALETTE[: len(selected_clusters)]
else:
    palette = categorical_palette(len(selected_clusters))

cluster_to_color = dict(zip(selected_clusters, palette))

# ---- 3) Auto point size by density (unless user-specified) ----
def auto_point_size(n_cells):
    if n_cells <= 50_000:
        return 2.0
    if n_cells <= 200_000:
        return 1.2
    if n_cells <= 1_000_000:
        return 0.6
    return 0.4

# ---- 4) Make output dir ----
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# ---- 5) Optionally filter experiments ----
exp_cats = adata.obs["experiment_id"].astype(str)
if EXPERIMENTS is None:
    experiments = sorted(exp_cats.unique())
else:
    experiments = [e for e in EXPERIMENTS if e in set(exp_cats.unique())]
    missing = [e for e in EXPERIMENTS if e not in set(exp_cats.unique())]
    if missing:
        print(f"[warn] Experiments not found and skipped: {missing}")

# ---- 6) Plot per experiment ----
for exp in experiments:
    mask_exp = (exp_cats == exp).values
    n_exp = int(mask_exp.sum())
    if n_exp == 0:
        continue

    xy = XY[mask_exp].astype(float, copy=True)
    if JITTER and JITTER > 0:
        xy += np.random.default_rng(0).normal(scale=JITTER, size=xy.shape)

    # Determine point size
    s = POINT_SIZE if POINT_SIZE is not None else auto_point_size(n_exp)

    # Prepare figure
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Only plot the selected clusters
    cls_exp = adata.obs.loc[mask_exp, CLUSTER_KEY].astype(str).to_numpy()

    handles = []
    labels = []
    for c in selected_clusters:
        cmask = (cls_exp == c)
        if not cmask.any():
            continue
        ax.scatter(
            xy[cmask, 0], xy[cmask, 1],
            s=s, c=cluster_to_color[c], alpha=ALPHA,
            edgecolors=EDGE_COLOR, linewidths=0.0, rasterized=True
        )
        handles.append(
            mpl.lines.Line2D([0],[0], marker="o", color="none",
                              markerfacecolor=cluster_to_color[c],
                              markersize=6, label=c)
        )
        labels.append(c)

    # Axes style
    ax.set_title(
        f"{exp}  •  n={n_exp:,} cells  •  {CLUSTER_KEY} (showing {len(selected_clusters)} selected)",
        fontsize=12
    )
    ax.set_xlabel(xy_names[0]); ax.set_ylabel(xy_names[1])

    # Flip Y if desired (image coordinate style)
    if FLIP_Y:
        ax.invert_yaxis()

    # Equal aspect so spatial geometry isn't distorted
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # Tidy ticks
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend outside
    if handles:
        ax.legend(
            handles=handles, labels=labels, title=f"{CLUSTER_KEY} (selected)",
            loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, ncol=1
        )

    # Tight layout & save
    fname = Path(OUTDIR) / f"spatial_clusters_selected__{exp}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {fname}")

print("Done.")
