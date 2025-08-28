#!/usr/bin/env python3
"""
Update `adata.obs["Histological Subtype"]` from a CSV mapping.

- Match CSV "Slide ID" to AnnData .obs["Patient ID"] (trimmed, case-insensitive).
- From CSV "Cancer subtype":
    * if text contains "Embryonal" -> "Embryonal RMS"
    * if text contains "Alveolar"  -> "Alveolar RMS"
- Only update overlapping/matched rows; everything else is unchanged.
- Skip Slide IDs with conflicting labels across CSV rows.
- Preserve/union categories for the subtype column.
- Backup original values to `Histological Subtype__orig` (created once).
- Writes `<input>_subtype_fix.h5ad`.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
from pandas.api.types import is_categorical_dtype

# ────────── paths & column names (edit these) ──────────
H5AD = "/projects/rubinstein-lab/USERS/domans/pediatric-sarcoma/sarcoma-features/ad_wsi.uni-2.h5ad"
CSV  = "MoreData/Metadata/MGH_metadata11082024.csv"

OBS_ID_COL      = "Patient ID"             # in adata.obs
OBS_SUBTYPE_COL = "Histological Subtype"   # in adata.obs

CSV_ID_COL      = "Slide ID"               # in CSV
CSV_SUBTYPE_COL = "Cancer subtype"         # in CSV

# ────────── helpers ──────────
def _keyify(x):
    """case-insensitive, trimmed string key"""
    return str(x).strip().lower()

def _std_subtype_from_text(txt: str):
    """Return standardized subtype label or None from free text."""
    if txt is None or (isinstance(txt, float) and np.isnan(txt)):
        return None
    low = str(txt).lower()
    has_emb = "embryonal" in low
    has_alv = "alveolar" in low
    if has_emb and not has_alv:
        return "Embryonal RMS"
    if has_alv and not has_emb:
        return "Alveolar RMS"
    # ambiguous (both or neither) -> None
    return None

# ────────── load inputs ──────────
adata = sc.read_h5ad(H5AD)
obs = adata.obs

missing_obs_cols = [c for c in (OBS_ID_COL, OBS_SUBTYPE_COL) if c not in obs.columns]
if missing_obs_cols:
    raise KeyError(f"Missing columns in adata.obs: {missing_obs_cols}")

csv = pd.read_csv(CSV)
missing_csv_cols = [c for c in (CSV_ID_COL, CSV_SUBTYPE_COL) if c not in csv.columns]
if missing_csv_cols:
    raise KeyError(f"Missing columns in CSV: {missing_csv_cols}")

# ────────── build CSV → standardized subtype mapping ──────────
df = csv[[CSV_ID_COL, CSV_SUBTYPE_COL]].copy()
df["__key"] = df[CSV_ID_COL].map(_keyify)
df["__std"] = df[CSV_SUBTYPE_COL].map(_std_subtype_from_text)

# keep rows that map cleanly to one of the standardized labels
df_valid = df.dropna(subset=["__std"]).copy()

# detect conflicts: same Slide ID mapped to both labels
conflict_keys = (
    df_valid.groupby("__key")["__std"]
    .nunique()
    .loc[lambda s: s > 1]
    .index
    .tolist()
)
if conflict_keys:
    print(f"[WARN] {len(conflict_keys)} Slide ID(s) have conflicting 'Cancer subtype' entries in CSV. "
          f"These will be skipped:\n  " + ", ".join(conflict_keys[:12]) + (" ..." if len(conflict_keys) > 12 else ""))

# drop conflicts and deduplicate to one label per key
df_map = (
    df_valid[~df_valid["__key"].isin(conflict_keys)]
    .drop_duplicates(subset="__key", keep="first")
    .set_index("__key")["__std"]
)

# ────────── prepare obs and apply updates (only overlaps) ──────────
# backup original once
backup_col = f"{OBS_SUBTYPE_COL}__orig"
if backup_col not in obs.columns:
    obs[backup_col] = obs[OBS_SUBTYPE_COL].copy()

# ensure categories include the two target labels (without duplicating/altering others)
col = obs[OBS_SUBTYPE_COL]
target = pd.Index(["Embryonal RMS", "Alveolar RMS"])
if is_categorical_dtype(col):
    new_cats = col.cat.categories.union(target)   # union preserves existing, adds missing
    obs[OBS_SUBTYPE_COL] = col.cat.set_categories(new_cats)

# build keys for obs and map new values from CSV (overlaps only)
obs["__key"] = obs[OBS_ID_COL].map(_keyify)
obs["__new_subtype"] = obs["__key"].map(df_map)  # None where no slide match (or conflicts)

# rows to update = overlaps with a clean mapping
update_mask = obs["__new_subtype"].notna()

# apply ONLY to overlaps
n_overlaps = int(update_mask.sum())
before_counts = obs[OBS_SUBTYPE_COL].value_counts(dropna=False)
obs.loc[update_mask, OBS_SUBTYPE_COL] = obs.loc[update_mask, "__new_subtype"]

# summarize actual changes (exclude cases where value was already equal)
changed_mask = update_mask & (obs[OBS_SUBTYPE_COL] != obs[backup_col])
n_changed = int(changed_mask.sum())

print(f"\nOverlaps matched (Patient ID ↔ Slide ID): {n_overlaps}")
print(f"Values actually changed in '{OBS_SUBTYPE_COL}': {n_changed}")
if n_overlaps != n_changed:
    print(f"Note: {n_overlaps - n_changed} overlapped row(s) already had the same value.")

print("\nNew subtype counts:")
print(obs[OBS_SUBTYPE_COL].value_counts(dropna=False))

if n_changed:
    print("\nExamples of changes (up to 10):")
    ex = obs.loc[changed_mask, [OBS_ID_COL, backup_col, OBS_SUBTYPE_COL]].head(10)
    with pd.option_context("display.max_colwidth", 80):
        print(ex)

# clean helper cols
obs.drop(columns=["__key", "__new_subtype"], inplace=True)

# ────────── write output ──────────
p = Path(H5AD)
out = p.with_name(p.stem + "_subtype_fix.h5ad")
adata.write(out)
print(f"\nWrote updated AnnData to: {out}")
