import anndata
import pandas as pd

# Load the .h5ad file
# Replace 'your_file.h5ad' with the actual path to your file
adata = anndata.read_h5ad('/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/sarcoma-features/ad_wsi.uni-2.h5ad')

# Examine the structure of the AnnData object

# 1. Overview of the object
print(adata)

# This will print a summary, including:
# - Number of observations (cells) and variables (genes)
# - Names of observation (obs) and variable (var) annotations
# - Names of unstructured (uns) annotations
# - Names of multi-dimensional observation (obsm) and variable (varm) annotations
# - Names of layers (additional data matrices)

# 2. Accessing the main data matrix
# adata.X contains the main data matrix (e.g., expression counts)
print("Data matrix shape:", adata.X.shape)
print("First 5 rows and 5 columns of the data matrix:\n", adata.X[:5, :5])

# 3. Accessing observation annotations (cell metadata)
# adata.obs is a Pandas DataFrame storing information about observations (cells)
print("\nObservation annotations (obs):\n", adata.obs.head())

# 4. Accessing variable annotations (gene metadata)
# adata.var is a Pandas DataFrame storing information about variables (genes)
print("\nVariable annotations (var):\n", adata.var.head())

# 5. Accessing unstructured annotations (general information)
# adata.uns is a dictionary-like object for storing unstructured information
print("\nUnstructured annotations (uns):\n", adata.uns.keys())
# You can access specific items like this:
# print(adata.uns['some_key'])

# 6. Accessing multi-dimensional observation annotations (e.g., embeddings)
# adata.obsm stores multi-dimensional annotations related to observations (cells),
# often including embeddings like UMAP or t-SNE coordinates.
print("\nMulti-dimensional observation annotations (obsm):\n", adata.obsm.keys())
# Accessing UMAP coordinates (if present)
# if 'X_umap' in adata.obsm:
#    print("UMAP coordinates shape:", adata.obsm['X_umap'].shape)
#    print("First 5 rows of UMAP coordinates:\n", adata.obsm['X_umap'][:5])

# 7. Accessing multi-dimensional variable annotations
# adata.varm stores multi-dimensional annotations related to variables (genes)
print("\nMulti-dimensional variable annotations (varm):\n", adata.varm.keys())

# 8. Accessing layers (additional data matrices)
# adata.layers stores additional matrices, for example, raw counts or normalized counts
print("\nLayers:\n", adata.layers.keys())
# Accessing a specific layer (if present)
# if 'spliced' in adata.layers:
#    print("Spliced layer shape:", adata.layers['spliced'].shape)
#    print("First 5 rows and 5 columns of the spliced layer:\n", adata.layers['spliced'][:5, :5])

import anndata
import pandas as pd

# Load the .h5ad file
# Replace 'your_file.h5ad' with the actual path to your file
adata = anndata.read_h5ad('/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/sarcoma-features/ad_wsi.uni-2.h5ad')

subtype_counts = adata.obs["Histological Subtype"].value_counts()

subtype_counts

# 1) How many observations come from MGH?
n_mgh = (adata.obs["Contributor"] == "MGH").sum()
print("Total MGH observations:", n_mgh)

# 2) Counts of Alveolar vs Embryonal within the MGH subset
mgh_counts = (
    adata.obs
         .loc[adata.obs["Contributor"] == "MGH", "Histological Subtype"]
         .value_counts()
)
print(mgh_counts)


# Required for Adobe Illustrator compatibility:
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42





