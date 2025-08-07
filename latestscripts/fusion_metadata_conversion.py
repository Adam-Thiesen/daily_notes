import pandas as pd

df = pd.read_csv("Metadata/Yale_mutations.csv")

df["Fusion status"] = (
    df["Molecular signature summary"]
      .str.lower()
      .map(
          lambda s: (
              "T" if isinstance(s, str) and any(k in s for k in ("fus pos", "fusion pos"))
              else "F" if isinstance(s, str) and any(k in s for k in ("fus neg", "fusion neg"))
              else "NA"
          )
      )
)

# save a fresh copy; don't include the DataFrame index
df.to_csv("yale_with_fusion_status.csv", index=False)

import pandas as pd

# ── 1.  read the CSV that already has Fusion status ──────────────────────────
#        (adjust the filename/path if you called it something else)
fusion_df = pd.read_csv("yale_with_fusion_status.csv",
                        usecols=["Slide ID", "Fusion status"])

# ── 2.  build a quick lookup dictionary: Slide ID → Fusion status ────────────
fusion_lookup = dict(zip(fusion_df["Slide ID"], fusion_df["Fusion status"]))

# ── 3.  add the new column to adata.obs, mapping by Slide ID ─────────────────
#        • .map() aligns each row’s Slide ID to the lookup
#        • missing IDs automatically become NaN, then we replace with "NA"
adata.obs["Fusion status"] = (
    adata.obs["Slide ID"].map(fusion_lookup).fillna("NA")
)

# (optional) make it a categorical column for cleaner plotting/storage
adata.obs["Fusion status"] = adata.obs["Fusion status"].astype("category")

print(adata.obs[["Slide ID", "Fusion status"]].head(50))

# ── 1.  read the second CSV ────────────────────────────────────────────────
#       Adjust the path / filename if needed.
fpfn_df = pd.read_csv(
    "Metadata/COG_mutations.csv",                 # ← your new file
    usecols=["Patient ID", "Grouping FP or FN"]
)

# ── 2.  build a mapping  Patient ID  →  'T' / 'F'  ─────────────────────────
status_lookup = (
    fpfn_df.set_index("Patient ID")                # use Patient ID as the key
           ["Grouping FP or FN"]                  # take the status column
           .str.upper()                           # make it case-insensitive
           .map({"FP": "T", "FN": "F"})           # convert to T / F
)  # Result: a Series whose index = Patient ID and values = 'T' / 'F' / NaN

# ── 3.  update adata.obs["Fusion status"] by matching Slide ID to Patient ID─
#       • .map() matches each Slide ID to status_lookup (NaN if no match)
#       • we only overwrite rows where a non-NaN status was found
new_vals = adata.obs["Slide ID"].map(status_lookup)

mask = new_vals.notna()                            # rows that actually matched
adata.obs.loc[mask, "Fusion status"] = new_vals[mask]

# ── 4.  optional: re-cast as category so plotting is tidy ──────────────────
adata.obs["Fusion status"] = adata.obs["Fusion status"].astype("category")

print("Fusion status updated from FP / FN table.")
print(adata.obs[["Slide ID", "Fusion status"]].head())


import pandas as pd

# --- 1.  pick up the slide identifier ------------------------------------
if "Slide ID" in adata.obs.columns:
    ids = adata.obs["Slide ID"].astype(str)
else:                                   # rare fallback: use the index
    ids = pd.Series(adata.obs.index.astype(str), index=adata.obs.index)

# keep only the part before ".oid…" (or any other '.' suffix)
slide_base = ids.str.split(".").str[0]

# --- 2.  build the lookup table ------------------------------------------
df = (
    pd.DataFrame({
        "SlideID": slide_base,
        "Fusion status": adata.obs["Fusion status"].astype(str)
    })
    .drop_duplicates("SlideID")         # one row per slide
)

# --- 3.  save -------------------------------------------------------------
df.to_csv("slides_with_fusion_status.csv", index=False)
print(f"Saved {len(df)} unique slide IDs → slides_with_fusion_status.csv")


