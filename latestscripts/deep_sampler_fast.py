#!/usr/bin/env python3

import os
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# 1) process_h5ad
##############################################################################
def process_h5ad(file_path):
    """
    Reads numeric feature data (X) plus metadata from obs,
    decodes as necessary, and returns (metadata_df, feature_df).
    Specifically handles 'Histological Subtype' as an AnnData
    categorical field (categories & codes).
    """
    with h5py.File(file_path, "r") as f:
        # -----------------------------
        # (A) METADATA
        # -----------------------------
        metadata = {}
        for key in f['obs'].keys():
            if isinstance(f[f'obs/{key}'], h5py.Dataset):
                metadata[key] = f[f'obs/{key}'][:]

        metadata_df = pd.DataFrame(metadata)

        # Decode "Patient ID"
        if 'Patient ID' in f['obs']:
            try:
                pid_obj = f['obs']['Patient ID']
                if (
                    isinstance(pid_obj, h5py.Group)
                    and 'categories' in pid_obj
                    and 'codes' in pid_obj
                ):
                    cat = pid_obj['categories'][:]
                    codes = pid_obj['codes'][:]
                    cat = [x.decode('utf-8') if isinstance(x, bytes) else x for x in cat]
                    metadata_df['Patient ID'] = [cat[c] for c in codes]
                else:
                    raw_data = pid_obj[:]
                    decoded = [x.decode('utf-8') if isinstance(x, bytes) else x for x in raw_data]
                    metadata_df['Patient ID'] = decoded
            except Exception as e:
                print(f"Could not process 'Patient ID': {e}")

        # Decode "Histological Subtype"
        if 'Histological Subtype' in f['obs']:
            try:
                hs_obj = f['obs']['Histological Subtype']
                if (
                    isinstance(hs_obj, h5py.Group)
                    and 'categories' in hs_obj
                    and 'codes' in hs_obj
                ):
                    cat = hs_obj['categories'][:]
                    codes = hs_obj['codes'][:]
                    cat = [x.decode('utf-8').strip() if isinstance(x, bytes) else x for x in cat]
                    subtypes = [cat[c] for c in codes]
                    subtype_map = {
                        'Embryonal RMS': 0,
                        'Embryonal': 0,
                        'Alveolar RMS': 1,
                        'Alveolar': 1
                    }
                    numeric_subtypes = [subtype_map.get(s, np.nan) for s in subtypes]
                    metadata_df['Histological Subtype'] = numeric_subtypes
                else:
                    raw_data = hs_obj[:]
                    subtypes = [x.decode('utf-8').strip() if isinstance(x, bytes) else x for x in raw_data]
                    subtype_map = {
                        'Embryonal RMS': 0,
                        'Embryonal': 0,
                        'Alveolar RMS': 1,
                        'Alveolar': 1
                    }
                    numeric_subtypes = [subtype_map.get(s, np.nan) for s in subtypes]
                    metadata_df['Histological Subtype'] = numeric_subtypes

            except Exception as e:
                print(f"Could not process 'Histological Subtype': {e}")

        # -----------------------------
        # (B) NUMERIC FEATURES: X
        # -----------------------------
        feature_data = f['X'][:]  # shape [n_samples, n_features]
        print(f"Feature data dtype for {file_path}: {feature_data.dtype}")

        if feature_data.dtype.kind == 'S':
            try:
                feature_data = np.array(
                    [[x.decode('utf-8') if isinstance(x, bytes) else x for x in row]
                     for row in feature_data],
                    dtype=str
                )
                feature_data = feature_data.astype(float)
            except Exception as e:
                print(f"Failed converting feature data to float: {e}")
                raise ValueError(f"Feature data in {file_path} contains non-numeric values.")

        try:
            feature_data = feature_data.astype(float)
        except Exception as e:
            raise ValueError(f"Feature data in {file_path} not numeric: {e}")

        var_names = f['var/_index'][:]
        var_names = [x.decode('utf-8') if isinstance(x, bytes) else x for x in var_names]

    feature_df = pd.DataFrame(feature_data, columns=var_names)

    # If Tissue ID is present, use it as index
    if 'Tissue ID' in metadata_df.columns:
        metadata_df['Tissue ID'] = (
            metadata_df['Tissue ID'].astype(str)
              .str.replace(r"^b'|'$", "", regex=True)
        )
        metadata_df.index = metadata_df['Tissue ID']
        feature_df.index = metadata_df.index

    return metadata_df, feature_df


##############################################################################
# 2) Combine metadata + features
##############################################################################
def load_h5ad_as_combined_df(file_path):
    """
    Reads h5ad using process_h5ad => returns a single combined DataFrame.
    """
    metadata_df, feature_df = process_h5ad(file_path)
    combined_df = pd.concat([metadata_df, feature_df], axis=1)
    return combined_df


##############################################################################
# 3) PyTorch Dataset
##############################################################################
class RhabdoH5adDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='Histological Subtype', group_col='Patient ID'):
        df = df.dropna(subset=[label_col])  # remove rows without label
        self.labels = df[label_col].astype(int).values
        self.features = df[feature_cols].values.astype(np.float32)

        # Group info
        if group_col in df.columns:
            self.groups = df[group_col].astype(str).values
        else:
            self.groups = np.array([f"grp_{i}" for i in range(len(df))])

        self.sample_ids = df.index.astype(str).tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],   dtype=torch.long)
        return x, y, self.sample_ids[idx], self.groups[idx]


##############################################################################
# 4) Simple MLP
##############################################################################
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # single logit
        )

    def forward(self, x):
        out = self.fc(x)
        return out.squeeze(1)  # => [batch_size]


##############################################################################
# 5) TRAIN & EVAL
##############################################################################
def train(model, dataloader, criterion, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels, _, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


def evaluate(model, dataloader):
    model.eval()
    device = next(model.parameters()).device

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, sample_ids, group_ids in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    bin_preds = (all_probs > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    acc = accuracy_score(all_labels, bin_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cr  = classification_report(all_labels, bin_preds, target_names=["Embryonal", "Alveolar"])

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC     : {auc:.4f}")
    print(cr)

    return acc, auc, all_probs, all_labels


##############################################################################
# 6) ADDITIONAL ANALYSIS: FIRST-LAYER WEIGHTS
##############################################################################
def analyze_first_layer_weights(model, fold_idx, percentile_block=10, outdir="."):
    """
    1. Grabs the MLP's first layer (model.fc[0]) weights & bias.
    2. Plots the overall distribution of flattened weights.
    3. Computes L1-based feature importance => sum of abs(weights) across hidden units
       => shape [input_dim].
    4. Plots distribution of L1 norms.
    5. Finds top 10% & bottom 10% => checks how many subfeatures from each base feature
       appear in each group. (Assumes each base feature has 'percentile_block' columns.)
    6. Writes a summary text file.
    """
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Layer is nn.Linear(input_dim, hidden_dim)
    # Weight shape => [hidden_dim, input_dim]
    layer = model.fc[0]
    W = layer.weight.detach().cpu().numpy()  # shape [hidden_dim, input_dim]
    b = layer.bias.detach().cpu().numpy()    # shape [hidden_dim]

    # Flatten W for distribution plotting
    W_flat = W.flatten()

    # Plot distribution of flattened weights
    plt.figure(figsize=(8,6))
    sns.histplot(W_flat, bins=50, kde=True)
    plt.title(f"First Layer Weight Distribution (Fold {fold_idx})")
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    outpath = os.path.join(outdir, f"uni_weights_distribution_fold_{fold_idx}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    # L1-based feature importance across hidden units => shape [input_dim]
    importance = np.sum(np.abs(W), axis=0)

    # Plot distribution of importance
    plt.figure(figsize=(8,6))
    sns.histplot(importance, bins=50, kde=True)
    plt.title(f"L1 Norm Distribution (Fold {fold_idx})")
    plt.xlabel("Sum of abs(weights) across hidden units")
    plt.ylabel("Count")
    outpath = os.path.join(outdir, f"uni_l1norm_distribution_fold_{fold_idx}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    # top 10% & bottom 10%
    n_input_features = importance.shape[0]
    top_k = int(np.ceil(0.1 * n_input_features))  # 10%
    sorted_idx = np.argsort(importance)[::-1]     # descending order
    top_10pct_idx = sorted_idx[:top_k]
    bottom_10pct_idx = sorted_idx[-top_k:]

    # If each base feature => 'percentile_block' columns
    # e.g. 768 base feats => each has 10 columns => total 7680
    top_features_count = {}
    for idx in top_10pct_idx:
        base_feat = idx // percentile_block
        top_features_count[base_feat] = top_features_count.get(base_feat, 0) + 1

    bottom_features_count = {}
    for idx in bottom_10pct_idx:
        base_feat = idx // percentile_block
        bottom_features_count[base_feat] = bottom_features_count.get(base_feat, 0) + 1

    # Write extended stats to a text file
    txt_out = os.path.join(outdir, f"uni_weight_analysis_fold_{fold_idx}.txt")
    with open(txt_out, "w") as f:
        f.write(f"=== Fold {fold_idx} Weight Analysis ===\n\n")
        f.write(f"Weight shape: {W.shape}\n")
        f.write(f"Bias shape  : {b.shape}\n\n")

        w_mean = np.mean(W_flat)
        w_std  = np.std(W_flat)
        w_min  = np.min(W_flat)
        w_max  = np.max(W_flat)
        f.write(f"Flattened W stats:\n")
        f.write(f"  Mean  : {w_mean:.6f}\n")
        f.write(f"  Std   : {w_std:.6f}\n")
        f.write(f"  Min   : {w_min:.6f}\n")
        f.write(f"  Max   : {w_max:.6f}\n\n")

        b_mean = np.mean(b)
        b_std  = np.std(b)
        b_min  = np.min(b)
        b_max  = np.max(b)
        f.write(f"Bias stats:\n")
        f.write(f"  Mean  : {b_mean:.6f}\n")
        f.write(f"  Std   : {b_std:.6f}\n")
        f.write(f"  Min   : {b_min:.6f}\n")
        f.write(f"  Max   : {b_max:.6f}\n\n")

        f.write(f"Number of input columns: {n_input_features}\n")
        f.write(f"Top 10% threshold: {top_k} columns\n")

        f.write("\n-- L1 importance: top 10% columns --\n")
        for i, col_idx in enumerate(top_10pct_idx[:20]):  # show top 20
            f.write(f"Rank {i+1}: col {col_idx} => L1={importance[col_idx]:.4f}\n")

        f.write("\n-- L1 importance: bottom 10% columns --\n")
        for i, col_idx in enumerate(bottom_10pct_idx[:20]): # show first 20
            f.write(f"Rank {n_input_features - top_k + i+1} from bottom: col {col_idx} => L1={importance[col_idx]:.4f}\n")

        f.write("\nBase feature -> # subfeatures in top 10%:\n")
        for bf, ct in sorted(top_features_count.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  Feature {bf}: {ct} subfeatures\n")

        f.write("\nBase feature -> # subfeatures in bottom 10%:\n")
        for bf, ct in sorted(bottom_features_count.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  Feature {bf}: {ct} subfeatures\n")

    print(f"[Fold {fold_idx}] Wrote weight analysis to {txt_out}")


##############################################################################
# 7) MAIN: CROSS-VALIDATION, + WEIGHT ANALYSIS
##############################################################################
def main(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (A) Load & Combine
    file_path = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.uni-2.h5ad"  # <--- Adjust
    combined_df = load_h5ad_as_combined_df(file_path)

    # Filter Tissue ID => only those ending in ".oid0"
    mask_oid = combined_df.index.str.endswith('.oid0')
    combined_df = combined_df[mask_oid].copy()
    print(f"After filtering .oid0, shape = {combined_df.shape}")

    # Filter subtypes => keep only 0/1
    if 'Histological Subtype' not in combined_df.columns:
        raise ValueError("No 'Histological Subtype' column found in combined_df!")
    combined_df = combined_df[combined_df['Histological Subtype'].isin([0,1])].copy()
    combined_df.dropna(subset=['Histological Subtype'], inplace=True)
    print(f"After filtering subtypes 0/1, shape = {combined_df.shape}")

    # (B) Select only numeric columns => automatically excludes "Path", etc.
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    label_col = "Histological Subtype"
    feature_cols = [c for c in numeric_cols if c != label_col]
    print(f"Feature columns (first 10 of {len(feature_cols)}): {feature_cols[:10]}")

    # (C) Create dataset
    dataset = RhabdoH5adDataset(
        df=combined_df,
        feature_cols=feature_cols,
        label_col=label_col,
        group_col='Patient ID'
    )

    # (D) Class weighting for BCE
    labels_array = dataset.labels
    class_counts = np.bincount(labels_array)
    minority_class = min(class_counts)
    majority_class = max(class_counts)
    pos_weight_value = majority_class / float(minority_class)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # (E) Cross-validation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    X_labels = labels_array
    groups = dataset.groups

    fold_results = []
    accum_cm = np.zeros((2,2), dtype=float)
    fold_idx = 1

    # If each base feature has 10 columns => percentile_block=10
    # Adjust if your data is different (e.g. no aggregator => percentile_block=1).
    percentile_block = 10

    # Make sure to create an output directory for plots
    outdir = "analysis_outputs"
    os.makedirs(outdir, exist_ok=True)

    for train_idx, test_idx in sgkf.split(
        np.zeros(len(X_labels)), 
        X_labels,
        groups=groups
    ):
        print(f"\n===== FOLD {fold_idx} =====")
        train_ds = Subset(dataset, train_idx)
        test_ds  = Subset(dataset, test_idx)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)

        # Create model
        input_dim = len(feature_cols)
        model = SimpleMLP(input_dim=input_dim, hidden_dim=256, dropout=0.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train
        train(model, train_loader, criterion, optimizer, device, epochs=50)

        # Evaluate
        acc, auc, probs, ytrue = evaluate(model, test_loader)
        fold_results.append((acc, auc))

        # Confusion matrix
        bin_preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(ytrue, bin_preds)
        accum_cm += cm

        # (F) Weight & L1-norm analysis
        analyze_first_layer_weights(
            model,
            fold_idx=fold_idx,
            percentile_block=percentile_block,
            outdir=outdir
        )

        fold_idx += 1

    # Final summary
    avg_acc = np.mean([r[0] for r in fold_results])
    avg_auc = np.mean([r[1] for r in fold_results])
    print("\n===== CROSS-VALIDATION SUMMARY =====")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average AUC     : {avg_auc:.4f}")

    # Plot average confusion matrix
    avg_cm = accum_cm / 5.0
    plt.figure(figsize=(5,4))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Embryonal (Pred)", "Alveolar (Pred)"],
                yticklabels=["Embryonal (True)", "Alveolar (True)"])
    plt.title("Average Confusion Matrix (5-Fold)")
    #plt.savefig(os.path.join(outdir, "uni_average_confusion_matrix.png"))
    #plt.close()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
