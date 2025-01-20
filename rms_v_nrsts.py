import seaborn as sns
import numpy as np
import pandas as pd
import h5py
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import scipy as sp
import os

#this one, then for top performer plot the ROC curve

#########################
# 1) process_h5ad
#########################
def process_h5ad(file_path):
    with h5py.File(file_path, "r") as f:
        # Access metadata
        metadata = {}
        for key in f['obs'].keys():
            # Check if the key points to a dataset
            if isinstance(f[f'obs/{key}'], h5py.Dataset):
                metadata[key] = f[f'obs/{key}'][:]
        
        # Convert metadata to DataFrame
        metadata_df = pd.DataFrame(metadata)

        # Check if 'Patient ID' exists with 'categories' and 'codes'
        if 'Patient ID' in f['obs']:
            try:
                categories = f['obs/Patient ID/categories'][:]
                codes = f['obs/Patient ID/codes'][:]
                # Decode categories if necessary
                categories = [
                    x.decode('utf-8') if isinstance(x, bytes) else x 
                    for x in categories
                ]
                metadata_df['Patient ID'] = [categories[code] for code in codes]
            except Exception as e:
                print(f"Could not process 'Patient ID': {e}")

        feature_data = f['X'][:]  # Extract feature matrix
        var_names = f['var/_index'][:].astype(str)  # Feature names (column names)

    # Convert feature data to a DataFrame
    feature_df = pd.DataFrame(feature_data, columns=var_names)

    if 'Tissue ID' in metadata_df.columns:
        # Remove b'...' from Tissue IDs
        metadata_df['Tissue ID'] = (
            metadata_df['Tissue ID']
            .astype(str)
            .str.replace(r"^b'|'$", "", regex=True)
        )
        # Use Tissue ID as index if you like:
        metadata_df.index = metadata_df['Tissue ID']
        feature_df.index = metadata_df.index

    return metadata_df, feature_df


#########################
# 2) extract_histological_type
#########################
def extract_histological_type(file_path):
    """Extracts 'Histological Type' from a .h5ad file."""
    try:
        with h5py.File(file_path, "r") as f:
            categories = f['obs/Histological Type/categories'][:]
            codes = f['obs/Histological Type/codes'][:]
            categories = [
                x.decode('utf-8') if isinstance(x, bytes) else x 
                for x in categories
            ]
            histological_type = [categories[code] for code in codes]
        return histological_type
    except Exception as e:
        print(f"Error extracting histological type from {file_path}: {e}")
        return []


#########################
# 2a) process_single_file
#########################
def process_single_file(file_path):
    """
    Reads one .h5ad file and returns a filtered DataFrame ready for logistic regression.
    Similar to process_combination, but for only one file.
    """
    print(f"\n[process_single_file] Processing file: {file_path}")

    # Step 1: process_h5ad (metadata, features)
    metadata, feature_df = process_h5ad(file_path)
    
    # For safety, check we have at least some samples
    if metadata.empty or feature_df.empty:
        print("   [Warning] No data found. Returning empty.")
        return pd.DataFrame()

    # Step 2: Extract histological subtype
    histological_type = extract_histological_type(file_path)
    if len(histological_type) != len(metadata):
        print("   [Warning] Histological subtype array length mismatch. Returning empty.")
        return pd.DataFrame()

    # Combine metadata + features
    merged_df = pd.concat([metadata, feature_df], axis=1)
    merged_df['Histological Type'] = histological_type
    # Step 1: Filter for specific 'Histological Type' values
    print(f"   merged_df shape before alveolar embryonal filter = {merged_df.shape}")

    filtered_df = merged_df[
        merged_df['Histological Type'].isin(['NRSTS', 'RMS'])
    ]
    print(f"   merged_df shape before Tissue ID filter = {filtered_df.shape}")
    # Step 2: Map 'Alveolar RMS' to 0 and 'Embryonal RMS' to 1
    filtered_df.loc[:, 'Histological Type'] = filtered_df['Histological Type'].map({
    'NRSTS': 0,
    'RMS': 1
    })
    

    print(f"   merged_df shape before Tissue ID filter = {filtered_df.shape}")

    # Optional Tissue ID filter
    filtered_df = filtered_df[filtered_df['Tissue ID'].str.endswith('.oid0')]
    print(f"   After .endswith('.oid0') filter: {filtered_df.shape[0]} rows remain.")

    if filtered_df.empty:
        print("[Error] No rows remain after Tissue ID filter. Returning empty.")
        return pd.DataFrame()

    #print("   First 5 Tissue IDs in filtered_df:", filtered_df['Tissue ID'].head().tolist())
    #print("   Histological Subtype counts:\n", filtered_df['Histological Subtype'].value_counts())

    return filtered_df


#########################
# 3) process_combination
#########################
def process_combination(file1, file2):
    """
    Assumes file1 and file2 have the SAME samples (same rows)
    but DIFFERENT sets of features (different columns).
    """
    print(f"\n[process_combination] Processing files:\n  1) {file1}\n  2) {file2}")

    metadata1, feature_df1 = process_h5ad(file1)
    metadata2, feature_df2 = process_h5ad(file2)

    common_index = metadata1.index.intersection(metadata2.index)
    print(f"   --> Found {len(common_index)} common samples.")

    if len(common_index) == 0:
        print("   [Warning] No matching samples between these two files. Returning empty.")
        return pd.DataFrame()

    metadata1 = metadata1.loc[common_index].copy()
    feature_df1 = feature_df1.loc[common_index].copy()
    metadata2 = metadata2.loc[common_index].copy()
    feature_df2 = feature_df2.loc[common_index].copy()

    combined_metadata = metadata1
    combined_features = pd.concat([feature_df1, feature_df2], axis=1)
    merged_df = pd.concat([combined_metadata, combined_features], axis=1)

    # Extract histological subtypes
    histological_type1 = extract_histological_type(file1)
    histological_type2 = extract_histological_type(file2)
    
    # Minimal check
    if len(histological_type1) != len(metadata1) or len(histological_type2) != len(metadata2):
        print("   [Warning] Histological subtype array length mismatch. Returning empty.")
        return pd.DataFrame()

    # Binarize 'Histological Subtype'
    merged_df['Histological Type'] = histological_type1
    # Step 1: Filter for specific 'Histological Type' values
    print(f"   merged_df shape before alveolar embryonal filter = {merged_df.shape}")
    filtered_df = merged_df[
        merged_df['Histological Type'].isin(['NRSTS', 'RMS'])
    ]
    print(f"   merged_df shape before Tissue ID filter = {filtered_df.shape}")
    # Step 2: Map 'Alveolar RMS' to 0 and 'Embryonal RMS' to 1
    filtered_df.loc[:, 'Histological Type'] = filtered_df['Histological Type'].map({
    'NRSTS': 0,
    'RMS': 1
    })


    filtered_df = filtered_df[filtered_df['Tissue ID'].str.endswith('.oid0')]
    print(f"   After .endswith('.oid0') filter: {filtered_df.shape[0]} rows remain.")

    if filtered_df.empty:
        raise ValueError(
            f"[Error] No rows with 'Tissue ID' ending in '.oid0' "
            f"in files: {file1}, {file2}"
        )

    print("   First 5 Tissue IDs in filtered_df:", filtered_df['Tissue ID'].head().tolist())
    print("   Histological Subtype counts:\n", filtered_df['Histological Type'].value_counts())

    return filtered_df


#########################
# 4) run_logistic_regression
#########################
def run_logistic_regression(filtered_df, num_iterations=1, splits=5):
    """Performs logistic regression with StratifiedGroupKFold."""
    try:
        print("\n[run_logistic_regression] Called.")
        #print("[Debug] filtered_df columns:", list(filtered_df.columns))
        #print("[Debug] filtered_df shape:", filtered_df.shape)

        # Adjust these indices as necessary for your data layout
        data_array = filtered_df.iloc[:, 3:-1].to_numpy()  # Features
        labels     = filtered_df.iloc[:, -1].to_numpy()    # Histological Subtype (0/1)
        groups     = filtered_df['Patient ID'].to_numpy()  # Grouping for SGKF

        print(f"[Debug] data_array shape = {data_array.shape}")
        print(f"[Debug] labels shape = {labels.shape}; unique labels = {np.unique(labels)}")

        aucs = []
        confusion_matrices = []
        all_metrics = []

        sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=None)

        fold_index = 0
        for iteration in range(num_iterations):
            print(f"  >> Iteration {iteration+1}/{num_iterations}")
            for train_idx, test_idx in sgkf.split(data_array, labels, groups=groups):
                fold_index += 1
                train_data, test_data = data_array[train_idx], data_array[test_idx]
                train_labels, test_labels = labels[train_idx], labels[test_idx]

                print(f"    Fold {fold_index}:")
                print(f"       train size = {len(train_idx)}, test size = {len(test_idx)}")
                print(f"       unique train labels = {np.unique(train_labels)}, unique test labels = {np.unique(test_labels)}")

                # Quick check for single-class test set
                if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
                    print("       [Warning] Single-class fold encountered. Skipping this fold entirely.")
                    continue 

                # Feature selection (t-test)
                try:
                    _, p_values = sp.stats.ttest_ind(
                        train_data[train_labels == 0],
                        train_data[train_labels == 1],
                        axis=0,
                        equal_var=False
                    )
                except ValueError as ve:
                    print(f"       [Error in t-test] {ve}")
                    continue

                selected_features = np.where(p_values < 0.05)[0]
                print(f"       {len(selected_features)} significant features found (p < 0.05).")

                if selected_features.size == 0:
                    print("       [Info] No features passed threshold. Skipping AUC for this fold.")
                    continue

                # Train logistic regression
                model = LR(
                    penalty='l1',
                    C=100,
                    class_weight='balanced',
                    solver='liblinear',
                    max_iter=2000
                )
                model.fit(train_data[:, selected_features], train_labels)

                # Predict
                probs = model.predict_proba(test_data[:, selected_features])[:, 1]
                preds = model.predict(test_data[:, selected_features])

                # Confusion matrix
                cm = confusion_matrix(test_labels, preds)
                confusion_matrices.append(cm)

                # Precision, recall, F1, accuracy
                precision = precision_score(test_labels, preds, zero_division=0)
                recall    = recall_score(test_labels, preds, zero_division=0)
                f1        = f1_score(test_labels, preds, zero_division=0)
                accuracy  = accuracy_score(test_labels, preds)

                all_metrics.append({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy
                })

                # Compute AUC if possible
                try:
                    fpr, tpr, _ = roc_curve(test_labels, probs)
                    fold_auc = auc(fpr, tpr)
                    aucs.append(fold_auc)
                    print(f"       [AUC for fold {fold_index}] = {fold_auc:.4f}")
                except ValueError as e:
                    print(f"       [AUC Error] {e}")

        # Summed confusion matrix
        summed_confusion_matrix = (
            np.sum(confusion_matrices, axis=0) if confusion_matrices else None
        )
        avg_metrics = pd.DataFrame(all_metrics).mean().to_dict() if all_metrics else {}

        # Final average AUC
        if len(aucs) > 0:
            final_auc = np.mean(aucs)
            print(f"[Debug] Collected {len(aucs)} AUCs. Final average AUC = {final_auc:.4f}")
        else:
            final_auc = None
            print("[Debug] No valid AUC values were collected.")

        return final_auc, summed_confusion_matrix, avg_metrics

    except Exception as e:
        print(f"[Error] Something went wrong in run_logistic_regression: {e}")
        return None, None, None


#########################
# 5) test_single_files
#########################
def test_single_files(backbones, file_ids):
    """
    Computes AUC for each single file. 
    Returns a 2D NumPy array (shape: len(file_ids) x len(backbones)).
    Rows = file IDs, Cols = backbones.
    """
    base_path = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/"
    n_files = len(file_ids)
    n_backbones = len(backbones)

    # Initialize with np.nan
    single_aucs = np.full((n_files, n_backbones), np.nan)

    for j, backbone in enumerate(backbones):
        for i, file_id in enumerate(file_ids):
            file_path = f"{base_path}ad_wsi.{backbone}-{file_id}.h5ad"
            print(f"\n[test_single_files] Single-file test => {backbone}-{file_id}")
            filtered_df = process_single_file(file_path)
            if not filtered_df.empty:
                auc_val, cm, metrics = run_logistic_regression(filtered_df, num_iterations=10, splits=5)
                if auc_val is not None:
                    single_aucs[i, j] = auc_val
                else:
                    print("  [Info] auc_val was None; skipping assignment.")
            else:
                print("  [Info] filtered_df was empty; skipping AUC assignment.")

    return single_aucs


#########################
# 6) test_combinations
#########################
def test_combinations(backbone, file_ids):
    base_path = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/"
    files = [f"{base_path}ad_wsi.{backbone}-{i}.h5ad" for i in file_ids]
    num_files = len(files)

    auc_matrix = np.full((num_files, num_files), -1, dtype=float)  # Diagonal = -1
    confusion_matrices = []
    metrics_list = []

    for i, j in combinations(range(num_files), 2):
        print(f"\n[test_combinations] Working on pair (i={i}, j={j}) => {files[i]} + {files[j]}")
        filtered_df = process_combination(files[i], files[j])
        if not filtered_df.empty:
            auc_val, cm, metrics = run_logistic_regression(filtered_df, num_iterations=10)
            if auc_val is not None:
                auc_matrix[i, j] = auc_val
                auc_matrix[j, i] = auc_val
                confusion_matrices.append(
                    (f"{backbone}-{file_ids[i]}", f"{backbone}-{file_ids[j]}", cm)
                )
                metrics_list.append(
                    (f"{backbone}-{file_ids[i]}", f"{backbone}-{file_ids[j]}", metrics)
                )
            else:
                print("  [Info] auc_val was None, so AUC not assigned.")
        else:
            print("  [Info] filtered_df was empty; skipping AUC assignment.")

    return auc_matrix, confusion_matrices, metrics_list


#########################
# 7) find_global_auc_min
#########################
def find_global_auc_min(single_aucs_matrix, combo_auc_dict):
    """
    Finds the minimum AUC across the single-file matrix
    AND all combination matrices for all backbones.
    combo_auc_dict is like {backbone: auc_matrix}.
    """
    all_aucs = []

    # Single-file
    all_aucs.extend(single_aucs_matrix.flatten())

    # Combo: replace -1 with NaN, gather valid entries
    for backbone, auc_mat in combo_auc_dict.items():
        auc_mat_valid = np.where(auc_mat == -1, np.nan, auc_mat)
        all_aucs.extend(auc_mat_valid.flatten())

    # Filter out NaNs
    all_aucs = [v for v in all_aucs if not np.isnan(v)]
    if len(all_aucs) == 0:
        return 0.0  # fallback if no AUC found
    return min(all_aucs)


#########################
# 8) plot_confusion_matrix
#########################
def plot_confusion_matrix(cm, label1, label2, output_dir):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {label1} vs {label2}")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"nrstsvrms_new_confusion_matrix_{label1}_vs_{label2}.png"), dpi=300)
    plt.close()


#########################
# 9) plot_single_aucs_heatmap
#########################
def plot_single_aucs_heatmap(
    single_aucs_matrix,
    file_ids,
    backbones,
    output_dir,
    vmin=0.5,
    vmax=1.0
):
    """
    Plots a heatmap for single-file AUCs.
    Rows = file IDs, Columns = backbones.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        single_aucs_matrix,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        xticklabels=[b.upper() for b in backbones],
        yticklabels=[str(fid) for fid in file_ids],
        cbar_kws={'label': 'AUC'},
        annot_kws={"size": 10, "weight": "bold"},
        vmin=vmin,
        vmax=vmax
    )
    plt.title("Single-File AUC Heatmap", fontsize=16, weight="bold")
    plt.xlabel("Backbone", fontsize=12, weight="bold")
    plt.ylabel("File ID", fontsize=12, weight="bold")
    plt.xticks(fontsize=10, rotation=45, weight="bold")
    plt.yticks(fontsize=10, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "nrstsvrms_new_heatmap_single_files.png"), dpi=300)
    plt.close()


#########################
# 10) plot_backbone_heatmap
#########################
def plot_auc_heatmap(
    auc_matrix,
    file_ids,
    backbone,
    output_dir,
    vmin=0.5,
    vmax=1.0
):
    labels = [f"{fid}" for fid in file_ids]

    # Convert -1 to np.nan
    auc_matrix = np.where(auc_matrix == -1, np.nan, auc_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        auc_matrix,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'AUC'},
        annot_kws={"size": 14, "weight": "bold"},
        vmin=vmin,
        vmax=vmax,
        mask=np.isnan(auc_matrix)
    )
    plt.title(f"AUROC Heatmap for {backbone.upper()}", fontsize=16, weight="bold")
    plt.xlabel("Field of view size 1", fontsize=12, weight="bold")
    plt.ylabel("Field of view size 2", fontsize=12, weight="bold")
    plt.xticks(fontsize=10, rotation=45, weight="bold")
    plt.yticks(fontsize=10, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"nrstsvrms_new_heatmap_{backbone}.png"), dpi=300)
    plt.close()

def plot_triangle_heatmap(
    auc_matrix,
    file_ids,
    backbone,
    output_dir,
    vmin=0.5,
    vmax=1.0
):
    """
    Plots the lower-triangular part of the combination AUC matrix
    (thus removing duplicate squares above the diagonal).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    labels = [str(fid) for fid in file_ids]

    # Convert -1 to np.nan (so these don't get plotted)
    auc_matrix = np.where(auc_matrix == -1, np.nan, auc_matrix)

    # Create a mask to hide the upper triangle. 
    # k=1 means we do NOT mask the diagonal. 
    # (If you also want to hide the diagonal, use k=0 instead.)
    mask = np.triu(np.ones_like(auc_matrix, dtype=bool), k=0)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        auc_matrix,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'AUC'},
        annot_kws={"size": 14, "weight": "bold"},
        vmin=vmin,
        vmax=vmax,
        mask=mask  # <-- This does the magic
    )

    plt.title(f"AUROC Heatmap (Triangle) for {backbone.upper()}", fontsize=16, weight="bold")
    plt.xlabel("File ID", fontsize=12, weight="bold")
    plt.ylabel("File ID", fontsize=12, weight="bold")
    plt.xticks(fontsize=10, rotation=45, weight="bold")
    plt.yticks(fontsize=10, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"tri_nrstsvrms_new_triangle_heatmap_{backbone}.png"), dpi=300)
    plt.close()



#########################
# Main block
#########################
if __name__ == "__main__":
    OUTPUT_DIR = "./output_heatmaps"
    BACKBONES = ['uni', 'conch', 'ctranspath', 'inception']
    FILE_IDS = [1, 2, 3, 4]

    # 1) Get single-file AUCs across all backbones and file IDs
    print("=== Testing SINGLE files across all backbones ===")
    single_aucs_matrix = test_single_files(BACKBONES, FILE_IDS)
    # shape: (len(FILE_IDS), len(BACKBONES))

    # 2) Get combination-file AUCs for each backbone
    print("\n=== Testing COMBINATIONS for each backbone ===")
    combo_auc_dict = {}  # {backbone: NxN AUC matrix}
    combo_confusion_dict = {}
    combo_metrics_dict = {}

    for backbone in BACKBONES:
        print(f"\n========== Combination testing for backbone: {backbone} ==========")
        auc_matrix, confusion_matrices, metrics_list = test_combinations(backbone, FILE_IDS)
        combo_auc_dict[backbone] = auc_matrix
        combo_confusion_dict[backbone] = confusion_matrices
        combo_metrics_dict[backbone] = metrics_list

    # 3) Determine the global min AUC (and we can fix the max at 1.0)
    global_min = find_global_auc_min(single_aucs_matrix, combo_auc_dict)
    global_max = 1.0
    print(f"\n[Info] Global AUC range across singles + combos: [{global_min:.3f}, {global_max:.3f}]")

    # 4) Plot single-file AUC heatmap with global_min, global_max
    print("\n=== Plotting SINGLE-FILE heatmap ===")
    plot_single_aucs_heatmap(
        single_aucs_matrix,
        FILE_IDS,
        BACKBONES,
        OUTPUT_DIR,
        vmin=global_min,
        vmax=global_max
    )

    # 5) Plot each backbone's combination heatmap with same scale
    print("\n=== Plotting COMBINATION-FILE heatmaps ===")
    for backbone in BACKBONES:
        plot_triangle_heatmap(
            combo_auc_dict[backbone],
            FILE_IDS,
            backbone,
            OUTPUT_DIR,
            vmin=global_min,
            vmax=global_max
        )

        # Also print confusion matrices and metrics
        # that were returned from test_combinations
        confusion_matrices = combo_confusion_dict[backbone]
        metrics_list = combo_metrics_dict[backbone]
        for (label1, label2, cm), (_, _, metrics) in zip(confusion_matrices, metrics_list):
            if cm is not None:
                plot_confusion_matrix(cm, label1, label2, OUTPUT_DIR)
            print(f"\nMetrics for {label1} vs {label2}:")
            for mkey, mval in metrics.items():
                print(f"  {mkey}: {mval:.3f}" if pd.notnull(mval) else f"  {mkey}: None")
