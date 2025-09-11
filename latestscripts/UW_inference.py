import pandas as pd

# Provide the path to your file below
file_path = "/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/UW-chenlab/results/01_HE.oid0/features/false-2-uni_features.tsv.gz"

# Read the compressed TSV file
df = pd.read_csv(file_path, sep="\t", compression="gzip")

# Display the first few rows
print(df.head())




# %% RMS vs NRSTS inference with numeric feature names matching training
import os
from pathlib import Path
import numpy as np
import pandas as pd

# model loading
import pickle
import joblib


# ───────────────────────── CONFIG ─────────────────────────
ROOT_RESULTS = Path("/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/UW-chenlab/results")  # <- set USER correctly
MODEL_PATH   = Path("rms_nrsts_clf.pkl")                              # <- your saved model (.pkl or .joblib)
OUT_CSV      = Path("rms_nrsts_inference.csv")

PERCENTILES  = list(range(5, 100, 10))   # [5,15,...,95]  (10 bins)
ORDER_20X_FIRST = False                  # training had uni-1 (40×) first

UNI2_20X_NAME = "false-2-uni_features.tsv.gz"  # 20×
UNI1_40X_NAME = "false-1-uni_features.tsv.gz"  # 40×

NON_FEATURE_COLS = {
    "barcode", "array_col", "array_row", "in_tissue",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
    "pxl_row_in_wsi", "pxl_col_in_wsi",
}
# ──────────────────────────

def load_model(path: Path):
    s = str(path).lower()
    if s.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return joblib.load(path)

def find_oid0_pairs(root: Path):
    pairs = []
    for dirpath, _, _ in os.walk(root):
        base = os.path.basename(dirpath)
        if not base.endswith(".oid0"):
            continue
        pid = base[:-5]
        fdir = Path(dirpath) / "features"
        if not fdir.is_dir():
            continue
        p20 = fdir / UNI2_20X_NAME
        p40 = fdir / UNI1_40X_NAME
        if p20.exists() and p40.exists():
            pairs.append((pid, p20, p40))
    return sorted(pairs, key=lambda x: x[0])

def detect_feature_columns(tsv_path: Path) -> list[str]:
    cols = pd.read_csv(tsv_path, nrows=0, compression="gzip").columns.tolist()
    feat_cols = [c for c in cols if c.startswith("feat_")]
    if not feat_cols:
        feat_cols = [c for c in cols if c not in NON_FEATURE_COLS]
    if not feat_cols:
        raise ValueError(f"No feature columns detected in {tsv_path}")
    return feat_cols

def compute_percentile_vector(tsv_path: Path, feat_cols: list[str], q: list[int]) -> np.ndarray:
    header = pd.read_csv(tsv_path, nrows=0, compression="gzip").columns.tolist()
    usecols = [c for c in feat_cols if c in header]
    if len(usecols) != len(feat_cols):
        missing = [c for c in feat_cols if c not in header]
        raise ValueError(f"{len(missing)} missing feature columns in {tsv_path} (e.g., {missing[:5]})")
    df = pd.read_csv(tsv_path, usecols=usecols, compression="gzip")
    df = df.reindex(columns=feat_cols)  # enforce canonical order
    arr = df.to_numpy(dtype=float, copy=False)
    # shape = (len(q), n_feats); ravel('C') => [p5 all feats] + [p15 all feats] + ...
    perc = np.nanpercentile(arr, q=q, axis=0)
    return perc.ravel(order="C")

def make_numeric_names(prefix: str, n_feats: int, n_percentiles: int) -> list[str]:
    """
    Names match training style: prefix_0 .. prefix_(n_feats*n_percentiles - 1),
    enumerated in the same order as ravel('C') above.
    """
    total = n_feats * n_percentiles
    return [f"{prefix}_{i}" for i in range(total)]

# 1) Discover patients
pairs = find_oid0_pairs(ROOT_RESULTS)
if not pairs:
    raise SystemExit(f"No .oid0 pairs with {UNI2_20X_NAME} and {UNI1_40X_NAME} under {ROOT_RESULTS}")

print(f"Found {len(pairs)} patients. Example: {pairs[0][0]}")

# 2) Canonical feature lists per magnification (from first pair)
_, first_20x, first_40x = pairs[0]
canon_20x_cols = detect_feature_columns(first_20x)  # 20×
canon_40x_cols = detect_feature_columns(first_40x)  # 40×
n20, n40 = len(canon_20x_cols), len(canon_40x_cols)
npct = len(PERCENTILES)
print(f"20× feature columns: {n20} | 40× feature columns: {n40}")

# Precompute column name templates matching training
names20 = make_numeric_names("uni-2", n20, npct)  # 20×
names40 = make_numeric_names("uni-1", n40, npct)  # 40×

# 3) Build per-patient rows
rows = []
pids = []
skipped = 0
for pid, path_20x, path_40x in pairs:
    try:
        vec20 = compute_percentile_vector(path_20x, canon_20x_cols, PERCENTILES)  # len = n20*npct
        vec40 = compute_percentile_vector(path_40x, canon_40x_cols, PERCENTILES)  # len = n40*npct
        if ORDER_20X_FIRST:
            vals  = np.concatenate([vec20, vec40], axis=0)
            names = names20 + names40
        else:
            vals  = np.concatenate([vec40, vec20], axis=0)
            names = names40 + names20
        rows.append(pd.Series(vals, index=names, dtype=float))
        pids.append(pid)
    except Exception as e:
        print(f"[WARN] Skipping {pid}: {e}")
        skipped += 1

if not rows:
    raise SystemExit("No valid feature vectors were produced; aborting.")
print(f"Prepared {len(rows)} patients; skipped {skipped}.")

X_df = pd.DataFrame(rows)
X_df.index = pids

# 4) Load model and align to training order if available
model = load_model(MODEL_PATH)

feature_order = getattr(model, "feature_order", None)
if feature_order is not None:
    missing = [c for c in feature_order if c not in X_df.columns]
    if missing:
        print("[ERROR] The following model feature columns were not found in the computed vectors "
              f"({len(missing)} missing). Example:", missing[:10])
        raise SystemExit("Feature alignment to model.feature_order failed.")
    X_aligned = X_df.reindex(columns=feature_order)
else:
    X_aligned = X_df  # fall back to constructed order

# Final dimension check
n_expected = getattr(model, "n_features_in_", None)
if n_expected is not None and X_aligned.shape[1] != n_expected:
    raise ValueError(
        f"Feature dimension mismatch: model expects {n_expected}, got {X_aligned.shape[1]}.\n"
        f"Check ORDER_20X_FIRST and the percentiles used."
    )

# 5) Predict
X_np = X_aligned.to_numpy()
y_pred = model.predict(X_np)

prob_nrsts = None
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_np)
    if hasattr(model, "classes_") and 1 in set(model.classes_):
        idx1 = int(np.where(model.classes_ == 1)[0][0])
        prob_nrsts = probs[:, idx1]

# 6) Save results
out = pd.DataFrame({"patient_id": X_aligned.index.tolist(),
                    "predicted_label": y_pred.astype(int)})
if prob_nrsts is not None:
    out["prob_NRSTS"] = prob_nrsts

out.to_csv(OUT_CSV, index=False)
print(f"Saved predictions to: {OUT_CSV.resolve()}")











import numpy as np
import pandas as pd
import h5py
import pickle
from sklearn.linear_model import LogisticRegression as LR


# ──────────────────────────────────────────────────────────────────────────────
# process_h5ad
# ──────────────────────────────────────────────────────────────────────────────
def process_h5ad(file_path, prefix=""):
    """
    Loads .h5ad data via h5py, returns (metadata_df, feature_df).
    Feature columns are renamed with `prefix` + "_" + original_name.
    """
    with h5py.File(file_path, "r") as f:
        # ---- obs (metadata)
        metadata = {}
        for key in f['obs'].keys():
            node = f[f'obs/{key}']
            if isinstance(node, h5py.Dataset):
                vals = node[:]
                # Best-effort byte decode for string-ish arrays
                if vals.dtype.kind in ("S", "O"):
                    try:
                        vals = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x for x in vals]
                    except Exception:
                        pass
                metadata[key] = vals
        metadata_df = pd.DataFrame(metadata)

        # ---- optional: Patient ID via categories/codes
        if 'Patient ID' in f['obs']:
            try:
                categories = f['obs/Patient ID/categories'][:]
                codes      = f['obs/Patient ID/codes'][:]
                categories = [x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x for x in categories]
                metadata_df['Patient ID'] = [categories[int(code)] for code in codes]
            except Exception as e:
                print(f"Could not process 'Patient ID': {e}")

        # ---- features matrix (X) and var names
        X = f['X'][:]
        var_raw = f['var/_index'][:]
        # robust decode for var names
        var_names = [
            v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v)
            for v in var_raw
        ]

    # prepend prefix to avoid collisions
    renamed_cols = [f"{prefix}_{col}" for col in var_names]
    feature_df = pd.DataFrame(X, columns=renamed_cols)

    # If Tissue ID exists, use as index to guarantee a shared join key
    if 'Tissue ID' in metadata_df.columns:
        metadata_df['Tissue ID'] = (
            metadata_df['Tissue ID'].astype(str)
            .str.replace(r"^b'|'$", "", regex=True)
        )
        metadata_df.index = metadata_df['Tissue ID']
        feature_df.index  = metadata_df.index

    return metadata_df, feature_df


# ──────────────────────────────────────────────────────────────────────────────
# extract_histological_type
# ──────────────────────────────────────────────────────────────────────────────
def extract_histological_type(file_path):
    """
    Returns list of histological type strings for the rows in file_path.
    Expected labels: 'Embryonal', 'Alveolar' (others will be filtered out).
    """
    try:
        with h5py.File(file_path, "r") as f:
            categories = f['obs/Histological Subtype/categories'][:]
            codes      = f['obs/Histological Subtype/codes'][:]
            categories = [x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x for x in categories]
            return [categories[int(code)] for code in codes]
    except Exception as e:
        print(f"Error extracting histological Subtype from {file_path}: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# process_combination
# ──────────────────────────────────────────────────────────────────────────────
def process_combination(file1, file2, enforce_oid0=False):
    """
    Merge two .h5ad files (same rows/samples, different columns/features).
    - Prefixes: 'uni-1' (40×) and 'uni-2' (20×)
    - Attach label column 'Histological Type' with values in {'RMS','NRSTS'}
      then map to {RMS:0, NRSTS:1}
    - Optionally enforce Tissue ID endswith '.oid0' to match inference.
    Returns merged DataFrame: [metadata..., features..., 'Histological Type'].
    """
    prefix1 = "uni-1"  # 40×
    prefix2 = "uni-2"  # 20×

    meta1, feat1 = process_h5ad(file1, prefix=prefix1)
    meta2, feat2 = process_h5ad(file2, prefix=prefix2)

    common_idx = meta1.index.intersection(meta2.index)
    if len(common_idx) == 0:
        print("No matching samples between the two .h5ad files.")
        return pd.DataFrame()

    meta1  = meta1.loc[common_idx].copy()
    feat1  = feat1.loc[common_idx].copy()
    meta2  = meta2.loc[common_idx].copy()
    feat2  = feat2.loc[common_idx].copy()

    # single meta (they should match)
    combined_meta = meta1
    combined_feat = pd.concat([feat1, feat2], axis=1)
    merged_df     = pd.concat([combined_meta, combined_feat], axis=1)

    # labels
    hist1 = extract_histological_type(file1)
    hist2 = extract_histological_type(file2)
    if (len(hist1) != len(meta1)) or (len(hist2) != len(meta2)):
        print("Histological type array mismatch. Check data.")
        return pd.DataFrame()
    merged_df['Histological Subtype'] = hist1  # rows align

    # optional restriction to '.oid0'
    if enforce_oid0 and 'Tissue ID' in merged_df.columns:
        merged_df['Tissue ID'] = merged_df['Tissue ID'].astype(str)
        merged_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]

    # keep only RMS / NRSTS and binarize
    merged_df = merged_df[merged_df['Histological Subtype'].isin(['Embryonal RMS', 'Alveolar RMS'])].copy()
    merged_df['Histological Subtype'] = merged_df['Histological Subtype'].map({'Embryonal RMS': 0, 'Alveolar RMS': 1})
    print(merged_df)

    return merged_df


# ──────────────────────────────────────────────────────────────────────────────
# train_final_model  (NO feature selection)
# ──────────────────────────────────────────────────────────────────────────────
def train_final_model(merged_df, output_clf_path="rms_nrsts_clf.pkl"):
    """
    Train logistic regression on ALL features (no statistical filtering).
    Saves a pickle with:
      - model coefficients/intercept
      - model.feature_order : list[str]  (exact training column order)
      - model.features      : {"uni-1": [...], "uni-2": [...]}
    """
    if merged_df.empty:
        raise ValueError("Empty dataframe provided to train_final_model.")

    # Select features robustly by prefix instead of relying on fixed positions
    feature_cols = [c for c in merged_df.columns if c.startswith("uni-1_") or c.startswith("uni-2_")]
    if not feature_cols:
        # fallback to positional slice if needed (metadata up to col 3; label at last col)
        feature_cols = merged_df.columns[3:-1].tolist()

    if 'Histological Subtype' not in merged_df.columns:
        raise ValueError("Expected 'Histological Subtype' column not found.")

    X = merged_df[feature_cols].to_numpy()
    y = merged_df['Histological Subtype'].to_numpy().astype(int)

    # Fit logistic regression on ALL features
    model = LR(
        penalty='l1',
        C=100,
        class_weight='balanced',
        solver='liblinear',
        max_iter=2000,
    )
    model.fit(X, y)

    # Attach helpful metadata to the model for consistent inference
    model.feature_order = feature_cols  # exact training order

    feat_dict = {"uni-1": [], "uni-2": []}
    for col in feature_cols:
        if col.startswith("uni-1_"):
            feat_dict["uni-1"].append(col.replace("uni-1_", "", 1))
        elif col.startswith("uni-2_"):
            feat_dict["uni-2"].append(col.replace("uni-2_", "", 1))
    model.features = feat_dict

    # Save model
    with open(output_clf_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained on ALL features and saved to: {output_clf_path}")
    print(f"Total features: {len(feature_cols)} | uni-1: {len(feat_dict['uni-1'])} | uni-2: {len(feat_dict['uni-2'])}")


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    file1 = "/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/sarcoma-features/ad_wsi.uni-1.h5ad"
    file2 = "/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/sarcoma-features/ad_wsi.uni-2.h5ad"

    df = process_combination(file1, file2, enforce_oid0=False)  # set True to mirror inference on .oid0
    if df.empty:
        print("No data after merging or mismatch encountered. Exiting.")
    else:
        train_final_model(df, output_clf_path="alveolar_embryonal_clf.pkl")




# %% RMS vs NRSTS inference with numeric feature names matching training
import os
from pathlib import Path
import numpy as np
import pandas as pd

# model loading
import pickle
import joblib


# ───────────────────────── CONFIG ─────────────────────────
ROOT_RESULTS = Path("/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/UW-chenlab/results")  # <- set USER correctly
MODEL_PATH   = Path("alveolar_embryonal_clf.pkl")                              # <- your saved model (.pkl or .joblib)
OUT_CSV      = Path("embryonal_alveolar_inference.csv")

PERCENTILES  = list(range(5, 100, 10))   # [5,15,...,95]  (10 bins)
ORDER_20X_FIRST = False                  # training had uni-1 (40×) first

UNI2_20X_NAME = "false-2-uni_features.tsv.gz"  # 20×
UNI1_40X_NAME = "false-1-uni_features.tsv.gz"  # 40×

NON_FEATURE_COLS = {
    "barcode", "array_col", "array_row", "in_tissue",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
    "pxl_row_in_wsi", "pxl_col_in_wsi",
}
# ──────────────────────────

def load_model(path: Path):
    s = str(path).lower()
    if s.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return joblib.load(path)

def find_oid0_pairs(root: Path):
    pairs = []
    for dirpath, _, _ in os.walk(root):
        base = os.path.basename(dirpath)
        if not base.endswith(".oid0"):
            continue
        pid = base[:-5]
        fdir = Path(dirpath) / "features"
        if not fdir.is_dir():
            continue
        p20 = fdir / UNI2_20X_NAME
        p40 = fdir / UNI1_40X_NAME
        if p20.exists() and p40.exists():
            pairs.append((pid, p20, p40))
    return sorted(pairs, key=lambda x: x[0])

def detect_feature_columns(tsv_path: Path) -> list[str]:
    cols = pd.read_csv(tsv_path, nrows=0, compression="gzip").columns.tolist()
    feat_cols = [c for c in cols if c.startswith("feat_")]
    if not feat_cols:
        feat_cols = [c for c in cols if c not in NON_FEATURE_COLS]
    if not feat_cols:
        raise ValueError(f"No feature columns detected in {tsv_path}")
    return feat_cols

def compute_percentile_vector(tsv_path: Path, feat_cols: list[str], q: list[int]) -> np.ndarray:
    header = pd.read_csv(tsv_path, nrows=0, compression="gzip").columns.tolist()
    usecols = [c for c in feat_cols if c in header]
    if len(usecols) != len(feat_cols):
        missing = [c for c in feat_cols if c not in header]
        raise ValueError(f"{len(missing)} missing feature columns in {tsv_path} (e.g., {missing[:5]})")
    df = pd.read_csv(tsv_path, usecols=usecols, compression="gzip")
    df = df.reindex(columns=feat_cols)  # enforce canonical order
    arr = df.to_numpy(dtype=float, copy=False)
    # shape = (len(q), n_feats); ravel('C') => [p5 all feats] + [p15 all feats] + ...
    perc = np.nanpercentile(arr, q=q, axis=0)
    return perc.ravel(order="C")

def make_numeric_names(prefix: str, n_feats: int, n_percentiles: int) -> list[str]:
    """
    Names match training style: prefix_0 .. prefix_(n_feats*n_percentiles - 1),
    enumerated in the same order as ravel('C') above.
    """
    total = n_feats * n_percentiles
    return [f"{prefix}_{i}" for i in range(total)]

# 1) Discover patients
pairs = find_oid0_pairs(ROOT_RESULTS)
if not pairs:
    raise SystemExit(f"No .oid0 pairs with {UNI2_20X_NAME} and {UNI1_40X_NAME} under {ROOT_RESULTS}")

print(f"Found {len(pairs)} patients. Example: {pairs[0][0]}")

# 2) Canonical feature lists per magnification (from first pair)
_, first_20x, first_40x = pairs[0]
canon_20x_cols = detect_feature_columns(first_20x)  # 20×
canon_40x_cols = detect_feature_columns(first_40x)  # 40×
n20, n40 = len(canon_20x_cols), len(canon_40x_cols)
npct = len(PERCENTILES)
print(f"20× feature columns: {n20} | 40× feature columns: {n40}")

# Precompute column name templates matching training
names20 = make_numeric_names("uni-2", n20, npct)  # 20×
names40 = make_numeric_names("uni-1", n40, npct)  # 40×

# 3) Build per-patient rows
rows = []
pids = []
skipped = 0
for pid, path_20x, path_40x in pairs:
    try:
        vec20 = compute_percentile_vector(path_20x, canon_20x_cols, PERCENTILES)  # len = n20*npct
        vec40 = compute_percentile_vector(path_40x, canon_40x_cols, PERCENTILES)  # len = n40*npct
        if ORDER_20X_FIRST:
            vals  = np.concatenate([vec20, vec40], axis=0)
            names = names20 + names40
        else:
            vals  = np.concatenate([vec40, vec20], axis=0)
            names = names40 + names20
        rows.append(pd.Series(vals, index=names, dtype=float))
        pids.append(pid)
    except Exception as e:
        print(f"[WARN] Skipping {pid}: {e}")
        skipped += 1

if not rows:
    raise SystemExit("No valid feature vectors were produced; aborting.")
print(f"Prepared {len(rows)} patients; skipped {skipped}.")

X_df = pd.DataFrame(rows)
X_df.index = pids

# 4) Load model and align to training order if available
model = load_model(MODEL_PATH)

feature_order = getattr(model, "feature_order", None)
if feature_order is not None:
    missing = [c for c in feature_order if c not in X_df.columns]
    if missing:
        print("[ERROR] The following model feature columns were not found in the computed vectors "
              f"({len(missing)} missing). Example:", missing[:10])
        raise SystemExit("Feature alignment to model.feature_order failed.")
    X_aligned = X_df.reindex(columns=feature_order)
else:
    X_aligned = X_df  # fall back to constructed order

# Final dimension check
n_expected = getattr(model, "n_features_in_", None)
if n_expected is not None and X_aligned.shape[1] != n_expected:
    raise ValueError(
        f"Feature dimension mismatch: model expects {n_expected}, got {X_aligned.shape[1]}.\n"
        f"Check ORDER_20X_FIRST and the percentiles used."
    )

# 5) Predict
X_np = X_aligned.to_numpy()
y_pred = model.predict(X_np)

prob_nrsts = None
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_np)
    if hasattr(model, "classes_") and 1 in set(model.classes_):
        idx1 = int(np.where(model.classes_ == 1)[0][0])
        prob_nrsts = probs[:, idx1]

# 6) Save results
out = pd.DataFrame({"patient_id": X_aligned.index.tolist(),
                    "predicted_label": y_pred.astype(int)})
if prob_nrsts is not None:
    out["prob_Alveolar"] = prob_nrsts

out.to_csv(OUT_CSV, index=False)
print(f"Saved predictions to: {OUT_CSV.resolve()}")








# %% UMAP (20× only) + Plotly interactive scatter with hover
import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

# Plotly
try:
    import plotly.express as px
except ImportError as e:
    raise SystemExit("Plotly not found. Install with:  python -m pip install --user plotly") from e

# ───────────────── config ─────────────────
ROOT_RESULTS   = Path("/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/UW-chenlab/results")  # ← set USER correctly
H5AD_UNI2_IN   = Path("/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/sarcoma-features/ad_wsi.uni-2.h5ad")  # existing 20× percentiles
H5AD_UNI2_OUT  = Path("uni2_percentiles_with_new.h5ad")               # updated uni-2 (old + new)
H5AD_COMBINED  = Path("uni2_all_for_umap.h5ad")                       # same as OUT (kept for clarity)

# Optional: predictions to join for coloring/hover (set to None to skip)
# Expected columns: at least ["predicted_label"]; optionally ["prob_NRSTS"].
# Index should be "sample_id" (e.g., "01_HE.oid0"). If you only have "patient_id",
# the code below will try to align via patient_id.
PREDICTIONS_CSV = None  # e.g., Path("rms_nrsts_inference.csv")

UMAP_CSV       = Path("false_umap_uni2_all.csv")
UMAP_HTML      = Path("false_umap_uni2_plotly.html")  # interactive HTML

# Percentiles used to summarize per-feature tile distributions
PERCENTILES    = list(range(5, 100, 10))  # [5,15,...,95] → 10 bins
UNI2_FILENAME  = "false-2-uni_features.tsv.gz"  # 20× features file in each .oid0/features/

# Only use .oid0 images
ENFORCE_OID0   = True

# UMAP knobs
DO_SCALE       = False   # set False to skip z-score scaling
USE_PCA        = False   # set False to build neighbors on raw X (cosine) without PCA
N_PCS          = 50
N_NEIGHBORS    = 15
MIN_DIST       = 0.3
RANDOM_SEED    = 0

# Plotly knobs
POINT_SIZE     = 10
OPACITY        = 0.9
TEMPLATE       = "plotly_white"
# ──────────────────────────────────

NON_FEATURE_COLS = {
    "barcode", "array_col", "array_row", "in_tissue",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
    "pxl_row_in_wsi", "pxl_col_in_wsi",
}

def _find_oid0_uni2(root: Path):
    """Return list of (sample_id, path_to_uni2_tsv) for all <PATIENT>.oid0 that have 20× TSV."""
    hits = []
    for dirpath, _, _ in os.walk(root):
        base = os.path.basename(dirpath)
        if ENFORCE_OID0 and not base.endswith(".oid0"):
            continue
        fdir = Path(dirpath) / "features"
        if fdir.is_dir():
            p20 = fdir / UNI2_FILENAME
            if p20.exists():
                hits.append((base, p20))  # sample_id like "01_HE.oid0"
    hits.sort(key=lambda x: x[0])
    return hits

def _detect_feat_cols(tsv_path: Path) -> list[str]:
    cols = pd.read_csv(tsv_path, nrows=0, compression="gzip").columns.tolist()
    feat_cols = [c for c in cols if c.startswith("feat_")]
    if not feat_cols:
        feat_cols = [c for c in cols if c not in NON_FEATURE_COLS]
    if not feat_cols:
        raise ValueError(f"No feature columns found in {tsv_path}")
    return feat_cols

def _percentile_vector(tsv_path: Path, feat_cols: list[str], percentiles: list[int]) -> np.ndarray:
    header = pd.read_csv(tsv_path, nrows=0, compression="gzip").columns.tolist()
    usecols = [c for c in feat_cols if c in header]
    if len(usecols) != len(feat_cols):
        missing = [c for c in feat_cols if c not in header]
        raise ValueError(f"{len(missing)} missing feature columns in {tsv_path} (e.g., {missing[:5]})")
    df = pd.read_csv(tsv_path, usecols=usecols, compression="gzip")
    df = df.reindex(columns=feat_cols)            # enforce canonical order
    arr = df.to_numpy(dtype=float, copy=False)
    perc = np.nanpercentile(arr, q=percentiles, axis=0)  # (len(q), n_feats)
    return perc.ravel(order="C")                           # len = len(q) * n_feats

# 1) Load existing uni-2 AnnData and (optionally) keep only .oid0 rows
print("Loading existing uni-2 AnnData…")
ad2 = sc.read_h5ad(H5AD_UNI2_IN)
if ENFORCE_OID0:
    mask = ad2.obs_names.astype(str).str.endswith(".oid0")
    ad2 = ad2[mask].copy()

# Prepare metadata columns
ad2.obs["source"] = "old"
ad2.obs["patient_id"] = ad2.obs_names.str.replace(r"\.oid0$", "", regex=True)

# 2) Find NEW uni-2 TSVs and build percentile vectors
pairs = _find_oid0_uni2(ROOT_RESULTS)
existing = set(ad2.obs_names.astype(str))
pairs_new = [(sid, p20) for sid, p20 in pairs if sid not in existing]

print(f"Existing rows: {ad2.n_obs} | New rows to add: {len(pairs_new)}")
X_new = None
idx_new = []

if pairs_new:
    # Establish canonical feature set from the first new TSV
    _, first_uni2 = pairs_new[0]
    feat_cols = _detect_feat_cols(first_uni2)
    n_feats   = len(feat_cols)
    n_bins    = len(PERCENTILES)

    # Sanity check: does the existing var length look like n_feats * n_bins?
    if ad2.n_vars != n_feats * n_bins:
        raise ValueError(
            f"Dimension mismatch:\n"
            f"- existing ad2.n_vars = {ad2.n_vars}\n"
            f"- computed n_feats*n_bins = {n_feats*n_bins} ({n_feats} feats × {n_bins} percentiles)\n"
            f"Check that PERCENTILES and per-tile feature count match how the .h5ad was built."
        )

    rows = []
    for sid, p20 in pairs_new:
        try:
            vec = _percentile_vector(p20, feat_cols, PERCENTILES)
            rows.append(vec)
            idx_new.append(sid)
        except Exception as e:
            print(f"[WARN] Skipping {sid}: {e}")

    if rows:
        X_new = np.vstack(rows).astype(np.float32)

# 3) Append new rows to the existing uni-2 AnnData
if X_new is not None and len(idx_new) > 0:
    ad_new = ad.AnnData(X_new)
    ad_new.var_names = ad2.var_names.copy()  # must match exactly
    ad_new.obs_names = pd.Index(idx_new, name=ad2.obs_names.name)
    ad_new.obs["source"] = "new"
    ad_new.obs["patient_id"] = ad_new.obs_names.str.replace(r"\.oid0$", "", regex=True)

    ad_all = ad.concat([ad2, ad_new], axis=0, join="outer", index_unique=None)
else:
    ad_all = ad2.copy()

print(f"Combined AnnData: {ad_all.n_obs} samples × {ad_all.n_vars} features")

# 4) Save updated uni-2 .h5ad (old + new)
ad_all.write(H5AD_UNI2_OUT)
ad_all.write(H5AD_COMBINED)  # same content; separate name for clarity
print(f"Wrote updated uni-2 AnnData → {H5AD_UNI2_OUT.resolve()}")

# 5) UMAP on ALL uni-2 samples (old + new)
adata = ad_all.copy()

# Optional scaling/PCA (explicit to avoid fallback warnings)
if DO_SCALE:
    sc.pp.scale(adata, zero_center=True, max_value=10)

if USE_PCA:
    sc.tl.pca(adata, n_comps=N_PCS, random_state=RANDOM_SEED)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, metric="cosine")
else:
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X", metric="cosine")

sc.tl.umap(adata, min_dist=MIN_DIST, random_state=RANDOM_SEED)

# UMAP coords → DataFrame
umap_df = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["UMAP1","UMAP2"])
umap_df["sample_id"]  = umap_df.index
umap_df["patient_id"] = adata.obs["patient_id"].astype(str).values
umap_df["source"]     = adata.obs["source"].astype(str).values

# (Optional) join predictions to the UMAP table
if PREDICTIONS_CSV is not None and Path(PREDICTIONS_CSV).exists():
    preds = pd.read_csv(PREDICTIONS_CSV)
    preds = preds.copy()
    # If the CSV has 'patient_id' but not 'sample_id', try to align by patient
    if "sample_id" in preds.columns:
        preds = preds.set_index("sample_id")
        umap_df = umap_df.join(preds, how="left")
    elif "patient_id" in preds.columns:
        umap_df = umap_df.merge(preds, on="patient_id", how="left")
    else:
        print("[WARN] Predictions CSV provided but no 'sample_id' or 'patient_id' column found; skipping join.")

# Save coordinates
umap_df.to_csv(UMAP_CSV, index=False)
print(f"Saved UMAP coords → {UMAP_CSV.resolve()}")

# Pick a sensible default color
color_by = "source"
if "predicted_label" in umap_df.columns:
    color_by = "predicted_label"  # override if predictions are present

# Build Plotly figure
hover_cols = ["sample_id", "patient_id", "source"]
if "predicted_label" in umap_df.columns:
    hover_cols += ["predicted_label"]
if "prob_NRSTS" in umap_df.columns:
    hover_cols += ["prob_NRSTS"]

fig = px.scatter(
    umap_df,
    x="UMAP1",
    y="UMAP2",
    color=color_by,
    hover_data=hover_cols,
    template=TEMPLATE,
    title=f"UMAP (uni-2) — colored by {color_by}",
)

fig.update_traces(
    marker=dict(size=POINT_SIZE, opacity=OPACITY),
    selector=dict(mode="markers")
)
fig.update_layout(
    legend_title_text=color_by,
    xaxis_title="UMAP1",
    yaxis_title="UMAP2",
)

# Show inline and save an interactive HTML
fig.show()
fig.write_html(UMAP_HTML, include_plotlyjs="cdn", full_html=True)
print(f"Saved interactive Plotly HTML → {UMAP_HTML.resolve()}")
