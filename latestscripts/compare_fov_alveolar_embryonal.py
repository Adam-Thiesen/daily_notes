import seaborn as sns
import numpy as np
import pandas as pd
import h5py
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
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
import scipy.stats as stats
import os

import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# The star of the show:
from statannotations.Annotator import Annotator

# Required for Adobe Illustrator compatibility:
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


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
    """Extracts 'Histological Subtype' from a .h5ad file."""
    try:
        with h5py.File(file_path, "r") as f:
            categories = f['obs/Histological Subtype/categories'][:]
            codes = f['obs/Histological Subtype/codes'][:]
            categories = [
                x.decode('utf-8') if isinstance(x, bytes) else x 
                for x in categories
            ]
            histological_type = [categories[code] for code in codes]
        return histological_type
    except Exception as e:
        #print(f"Error extracting histological type from {file_path}: {e}")
        return []


#########################
# 2a) process_single_file
#########################
def process_single_file(file_path):
    """
    Reads one .h5ad file and returns a filtered DataFrame ready for logistic regression.
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
        #print("   [Warning] Histological subtype array length mismatch. Returning empty.")
        return pd.DataFrame()

    # Combine metadata + features
    merged_df = pd.concat([metadata, feature_df], axis=1)
    merged_df['Histological Subtype'] = histological_type

    #print(f"   merged_df shape before alveolar embryonal filter = {merged_df.shape}")

    filtered_df = merged_df[
        merged_df['Histological Subtype'].isin(['Alveolar RMS', 'Embryonal RMS'])
    ]
    #print(f"   merged_df shape before Tissue ID filter = {filtered_df.shape}")

    filtered_df.loc[:, 'Histological Subtype'] = filtered_df['Histological Subtype'].map({
        'Alveolar RMS': 0,
        'Embryonal RMS': 1
    })

    #print(f"   merged_df shape before Tissue ID filter = {filtered_df.shape}")

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
    merged_df['Histological Subtype'] = histological_type1

    #print(f"   merged_df shape before alveolar embryonal filter = {merged_df.shape}")
    filtered_df = merged_df[
        merged_df['Histological Subtype'].isin(['Alveolar RMS', 'Embryonal RMS'])
    ]
    #print(f"   merged_df shape before Tissue ID filter = {filtered_df.shape}")

    filtered_df.loc[:, 'Histological Subtype'] = filtered_df['Histological Subtype'].map({
        'Alveolar RMS': 0,
        'Embryonal RMS': 1
    })

    filtered_df = filtered_df[filtered_df['Tissue ID'].str.endswith('.oid0')]
    #print(f"   After .endswith('.oid0') filter: {filtered_df.shape[0]} rows remain.")

    if filtered_df.empty:
        raise ValueError(
            f"[Error] No rows with 'Tissue ID' ending in '.oid0' "
            f"in files: {file1}, {file2}"
        )

    #print("   First 5 Tissue IDs in filtered_df:", filtered_df['Tissue ID'].head().tolist())
    #print("   Histological Subtype counts:\n", filtered_df['Histological Subtype'].value_counts())

    return filtered_df


#########################
# 4) run_logistic_regression
#########################
def run_logistic_regression(filtered_df, num_iterations=1, splits=5):
    """
    Performs logistic regression with StratifiedGroupKFold.
    Returns:
      final_auc (float),
      final_std (float),
      summed_confusion_matrix (2D np.array),
      avg_metrics (dict),
      all_fold_aucs (list of float) - each fold's AUC
    """
    try:
        #print("\n[run_logistic_regression] Called.")

        data_array = filtered_df.iloc[:, 3:-1].to_numpy()  # Features
        labels     = filtered_df.iloc[:, -1].to_numpy()    # Histological Subtype (0/1)
        groups     = filtered_df['Patient ID'].to_numpy()  # Grouping

        #print(f"[Debug] data_array shape = {data_array.shape}")
        #print(f"[Debug] labels shape = {labels.shape}; unique labels = {np.unique(labels)}")

        aucs = []
        confusion_matrices = []
        all_metrics = []

        sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=None)
        fold_index = 0

        for iteration in range(num_iterations):
            #print(f"  >> Iteration {iteration+1}/{num_iterations}")
            for train_idx, test_idx in sgkf.split(data_array, labels, groups=groups):
                fold_index += 1
                train_data, test_data = data_array[train_idx], data_array[test_idx]
                train_labels, test_labels = labels[train_idx], labels[test_idx]

                #print(f"    Fold {fold_index}:")
                #print(f"       train size = {len(train_idx)}, test size = {len(test_idx)}")
                #print(f"       unique train labels = {np.unique(train_labels)}, unique test labels = {np.unique(test_labels)}")

                # Quick check for single-class fold
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

                # Compute AUC
                try:
                    fpr, tpr, _ = roc_curve(test_labels, probs)
                    fold_auc = auc(fpr, tpr)
                    aucs.append(fold_auc)
                    #print(f"       [AUC for fold {fold_index}] = {fold_auc:.4f}")
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
            final_std = np.std(aucs, ddof=1)
            #print(f"[Debug] Collected {len(aucs)} AUCs. Final average AUC = {final_auc:.4f}; std = {final_std:.4f}")
        else:
            final_auc = None
            final_std = None
            #print("[Debug] No valid AUC values were collected.")

        return final_auc, final_std, summed_confusion_matrix, avg_metrics, aucs

    except Exception as e:
        #rint(f"[Error] Something went wrong in run_logistic_regression: {e}")
        return None, None, None, None, []


#########################
# 5) test_single_files
#########################
def test_single_files(backbones, file_ids):
    """
    Computes AUC for each single file. 
    Returns:
      single_aucs (mean)   -> shape (len(file_ids), len(backbones))
      single_aucs_stds     -> shape (len(file_ids), len(backbones))
      single_aucs_distribution -> dict {(backbone, file_id): [fold_aucs]}
    """
    base_path = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/"
    n_files = len(file_ids)
    n_backbones = len(backbones)

    single_aucs = np.full((n_files, n_backbones), np.nan)
    single_aucs_stds = np.full((n_files, n_backbones), np.nan)

    # We will store the raw distributions here
    single_aucs_distribution = {}  # dict key: (backbone, file_id) -> list of fold AUCs

    for j, backbone in enumerate(backbones):
        for i, file_id in enumerate(file_ids):
            file_path = f"{base_path}ad_wsi.{backbone}-{file_id}.h5ad"
            #print(f"\n[test_single_files] Single-file test => {backbone}-{file_id}")
            filtered_df = process_single_file(file_path)
            if not filtered_df.empty:
                mean_auc, std_auc, cm, metrics, fold_aucs = run_logistic_regression(
                    filtered_df, 
                    num_iterations=10, 
                    splits=5
                )
                single_aucs_distribution[(backbone, file_id)] = fold_aucs

                if mean_auc is not None:
                    single_aucs[i, j] = mean_auc
                    single_aucs_stds[i, j] = std_auc
                else:
                    print("  [Info] mean_auc was None; skipping assignment.")
            else:
                print("  [Info] filtered_df was empty; skipping assignment.")
                single_aucs_distribution[(backbone, file_id)] = []

    return single_aucs, single_aucs_stds, single_aucs_distribution


#########################
# 6) test_combinations
#########################
def test_combinations(backbone, file_ids):
    """
    For each pair (i,j), runs logistic regression on combined features,
    returning two NxN arrays: mean_AUC and std_AUC.
    """
    base_path = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/"
    files = [f"{base_path}ad_wsi.{backbone}-{i}.h5ad" for i in file_ids]
    num_files = len(files)

    auc_matrix = np.full((num_files, num_files), -1.0, dtype=float)
    std_matrix = np.full((num_files, num_files), -1.0, dtype=float)

    confusion_matrices = []
    metrics_list = []

    # For each combination
    for i, j in combinations(range(num_files), 2):
        #print(f"\n[test_combinations] Working on pair (i={i}, j={j}) => {files[i]} + {files[j]}")
        filtered_df = process_combination(files[i], files[j])
        if not filtered_df.empty:
            mean_auc, std_auc, cm, metrics, _ = run_logistic_regression(
                filtered_df, 
                num_iterations=10
            )
            if mean_auc is not None:
                # Symmetrically assign
                auc_matrix[i, j] = mean_auc
                auc_matrix[j, i] = mean_auc

                std_matrix[i, j] = std_auc
                std_matrix[j, i] = std_auc

                confusion_matrices.append(
                    (f"{backbone}-{file_ids[i]}", f"{backbone}-{file_ids[j]}", cm)
                )
                metrics_list.append(
                    (f"{backbone}-{file_ids[i]}", f"{backbone}-{file_ids[j]}", metrics)
                )
            else:
                print("  [Info] mean_auc was None, so no assignment.")
        else:
            print("  [Info] filtered_df was empty; skipping assignment.")

    return auc_matrix, std_matrix, confusion_matrices, metrics_list


#########################
# 7) find_global_auc_min
#########################
def find_global_auc_min(single_aucs_matrix, combo_auc_dict):
    """
    Finds the minimum AUC across the single-file matrix
    AND all combination matrices for all backbones.
    combo_auc_dict is like {backbone: (auc_matrix, std_matrix)}.
    """
    all_aucs = []

    # Single-file
    all_aucs.extend(single_aucs_matrix.flatten())

    # Combo: replace -1 with NaN, gather valid entries
    for backbone, (auc_mat, _) in combo_auc_dict.items():
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
    plt.savefig(os.path.join(output_dir, f"ae_new_confusion_matrix_{label1}_vs_{label2}.pdf"), dpi=300, transparent=True)
    plt.close()


#########################
# 9) plot_single_aucs_heatmap
#########################
def plot_single_aucs_heatmap(
    single_means,
    single_stds,
    file_ids,
    backbones,
    output_dir,
    vmin=0.5,
    vmax=1.0
):
    """
    Plots a heatmap for single-file AUCs.
    Rows = file IDs, Columns = backbones.
    - single_means: 2D (n_files x n_backbones)
    - single_stds: 2D (n_files x n_backbones)
    """
    n_files, n_backbones = single_means.shape

    # Build custom annotation array: "mean\n(±std)"
    annot = np.empty_like(single_means, dtype=object)
    for i in range(n_files):
        for j in range(n_backbones):
            mean_val = single_means[i, j]
            std_val  = single_stds[i, j]
            if np.isnan(mean_val):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{mean_val:.3f}\n± {std_val:.3f}"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        single_means,
        annot=annot,
        fmt="",
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
    plt.savefig(os.path.join(output_dir, "ae_new2_heatmap_single_files.pdf"), dpi=300, transparent=True)
    plt.close()


#########################
# 10) plot_triangle_heatmap_with_light_grey_diag
#########################
import matplotlib.colors as mcolors

def plot_triangle_heatmap_with_light_grey_diag(
    mean_matrix,
    std_matrix,
    file_ids,
    backbone,
    output_dir,
    vmin=0.5,
    vmax=1.0
):
    """
    Plots the lower-triangular heatmap of `mean_matrix` in Reds,
    overlaying diagonal squares in a lighter grey.
    Annotations show "mean\n± std" for the lower triangle.
    """
    # Convert -1 to NaN for invalid combos
    mean_matrix = np.where(mean_matrix == -1, np.nan, mean_matrix)
    std_matrix = np.where(std_matrix == -1, np.nan, std_matrix)
    n = mean_matrix.shape[0]

    # Build custom annotation array
    annot = np.empty_like(mean_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal => blank
                annot[i, j] = ""
            elif i < j:
                # Upper triangle => blank
                annot[i, j] = ""
            else:
                # Lower triangle => "mean\n± std"
                val_m = mean_matrix[i, j]
                val_s = std_matrix[i, j]
                if np.isnan(val_m):
                    annot[i, j] = ""
                else:
                    annot[i, j] = f"{val_m:.3f}\n± {val_s:.3f}"

    labels = [str(fid) for fid in file_ids]

    # Mask for upper triangle
    lower_tri_mask = np.triu(np.ones_like(mean_matrix, dtype=bool), k=1)

    # Diagonal mask
    diag_mask = ~np.eye(n, dtype=bool)

    # 1) Plot lower triangle
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        mean_matrix,
        mask=lower_tri_mask,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt="",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'AUC'},
        annot_kws={"size": 9, "weight": "bold"},
        alpha=1.0,
        zorder=1
    )

    # 2) Overlay diagonal squares in grey
    diag_vals = np.diag(mean_matrix).copy()
    np.fill_diagonal(mean_matrix, 999.0)  # sentinel
    light_grey_cmap = mcolors.ListedColormap(["#DDDDDD"])
    norm = mcolors.Normalize(vmin=998.5, vmax=999.5)

    sns.heatmap(
        mean_matrix,
        mask=diag_mask,       # Hide everything except diagonal
        cmap=light_grey_cmap,
        norm=norm,
        cbar=False,
        annot=False,
        alpha=1.0,
        zorder=2
    )

    # Restore original diagonal
    for i in range(n):
        mean_matrix[i, i] = diag_vals[i]

    plt.title(f"Combination AUC for {backbone.upper()}",
              fontsize=14, weight="bold")
    plt.xlabel("Field of View", fontsize=12, weight="bold")
    plt.ylabel("Field of View", fontsize=12, weight="bold")
    plt.xticks(fontsize=10, rotation=45, weight="bold")
    plt.yticks(fontsize=10, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"ae_triangle_heatmap_lightgrey_diag_{backbone}.pdf"), dpi=300, transparent=True)
    plt.close()


#########################
# 11) plot_boxplot_and_compute_pvals
#########################
def plot_boxplot_and_compute_pvals(
    single_aucs_distribution,
    backbone,
    file_ids,
    output_dir
):
    """
    1) Creates a Pandas DataFrame of all per-fold AUCs for the given backbone,
       with columns ['FileID', 'AUC'].
    2) Plots a boxplot with x=FileID, y=AUC.
    3) Computes pairwise t-test p-values for each (FileID1, FileID2) pair.
    4) Adds significance bars and asterisks to the plot where applicable.
    """

    # Build the DataFrame
    rows = []
    for fid in file_ids:
        fold_aucs = single_aucs_distribution.get((backbone, fid), [])
        for val in fold_aucs:
            rows.append({"FileID": fid, "AUC": val})
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[Warning] No AUC distribution data found for backbone={backbone}. Skipping box plot.")
        return

    # --- Create the plot ---
    plt.figure(figsize=(6, 5))
    ax = plt.gca()  # We'll explicitly grab the current Axes

    # Basic boxplot + stripplot
    sns.boxplot(data=df, x="FileID", y="AUC", color="lightblue", ax=ax)
    sns.stripplot(data=df, x="FileID", y="AUC", color="black", alpha=0.7, jitter=True, ax=ax)

    plt.title(f"{backbone.upper()} Single-File AUC Distribution", fontsize=14, weight="bold")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Field of View (File ID)", fontsize=12, weight="bold")
    plt.ylabel("AUC", fontsize=12, weight="bold")
    plt.tight_layout()

    # --- Add significance brackets ---
    # Build all pairwise comparisons: e.g. [('1', '2'), ('1', '3'), ('2', '3'), ...]
    # Make sure your file_ids are strings if your x-axis is string-based
    # (or ensure the pairs match what's in your DataFrame exactly)
    pairs = list(itertools.combinations(file_ids, 2))
    print(pairs)

    # Create an Annotator object
    annot = Annotator(ax, pairs, data=df, x="FileID", y="AUC")

    # Configure to use a t-test, show significance as stars, place brackets "outside"
    # (You can tune offsets to avoid bracket overlap.)
    annot.configure(
        test='t-test_ind',
        text_format='star',
        loc='inside',       # or 'center'
        line_offset=0.05,   # smaller offset
        verbose=2)

    annot.apply_and_annotate()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"boxplot_single_{backbone}.pdf"), dpi=300, transparent=True)
    plt.close()

    # --- (Optional) Print out p-values in the console ---
    print(f"\n=== Pairwise p-values for {backbone.upper()} ===")
    for i in range(len(file_ids)):
        for j in range(i+1, len(file_ids)):
            fid1 = file_ids[i]
            fid2 = file_ids[j]
            dist1 = df.loc[df["FileID"] == fid1, "AUC"].values
            dist2 = df.loc[df["FileID"] == fid2, "AUC"].values
            # Basic two-sided t-test
            tstat, pval = stats.ttest_ind(dist1, dist2, equal_var=False)
            print(f"  Compare FOV {fid1} vs {fid2}: t={tstat:.3f}, p={pval:.4g}")


#########################
# 12) compare_backbones_for_fov
#########################
def compare_backbones_for_fov(
    single_aucs_distribution,
    backbones,
    fov=2,
    output_dir="./"
):
    """
    1) Collect all fold-level AUCs for the specified FOV=2 (or any given fov).
    2) Plot a bar chart (with SD error bars) comparing each backbone.
    3) Perform pairwise t-tests between each pair of backbones.
    4) Add significance bars (asterisks) above bars where possible.
    """
    # Build a DataFrame with columns: ['Backbone', 'AUC']
    rows = []
    for backbone in backbones:
        dist = single_aucs_distribution.get((backbone, fov), [])
        for auc_val in dist:
            rows.append({"Backbone": backbone, "AUC": auc_val})
    df = pd.DataFrame(rows)

    if df.empty:
        print(f"[Warning] No fold-level data found for FOV={fov}. Skipping bar chart.")
        return

    # Compute means, stds per backbone
    agg_df = df.groupby("Backbone")["AUC"].agg(["mean", "std"]).reset_index()

    # --- Create the bar plot ---
    plt.figure(figsize=(6, 5))
    ax = plt.gca()  # current Axes

    sns.barplot(
        data=df,
        x="Backbone",
        y="AUC",
        ci="sd",
        color="lightblue",
        capsize=0.2,
        errcolor="black",
        ax=ax
    )
    sns.stripplot(data=df, x="Backbone", y="AUC", color="black", alpha=0.7, jitter=True, ax=ax)

    plt.title(f"Comparison of Backbones at FOV={fov}", fontsize=14, weight="bold")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Backbone", fontsize=12, weight="bold")
    plt.ylabel("AUC", fontsize=12, weight="bold")
    plt.tight_layout()

    # --- Add significance annotations ---
    pairs = list(itertools.combinations(backbones, 2))
    print(pairs)
    annot = Annotator(ax, pairs, data=df, x="Backbone", y="AUC")
    annot.configure(
        test='t-test_ind',
        text_format='star',
        loc='inside',       # or 'center'
        line_offset=0.05,   # smaller offset
        verbose=2)

    annot.apply_and_annotate()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"barplot_backbones_FOV{fov}.pdf"), dpi=300, transparent=True)
    plt.close()

    # --- (Optional) Print out pairwise t-tests in the console ---
    print(f"\n=== Pairwise p-values for FOV={fov} across backbones ===")
    for i in range(len(backbones)):
        for j in range(i+1, len(backbones)):
            b1 = backbones[i]
            b2 = backbones[j]
            dist1 = df.loc[df["Backbone"] == b1, "AUC"].values
            dist2 = df.loc[df["Backbone"] == b2, "AUC"].values
            tstat, pval = stats.ttest_ind(dist1, dist2, equal_var=False)
            print(f"  Compare {b1} vs {b2}: t={tstat:.3f}, p={pval:.4g}")


#########################
# Main block
#########################
if __name__ == "__main__":
    OUTPUT_DIR = "./output_heatmaps"
    BACKBONES = ['uni', 'conch', 'ctranspath', 'inception']
    FILE_IDS = [1, 2, 3, 4]

    # 1) SINGLE-FILE tests
    print("=== Testing SINGLE files across all backbones ===")
    single_aucs_matrix, single_aucs_stds_matrix, single_aucs_distribution = test_single_files(BACKBONES, FILE_IDS)

    # 2) COMBINATION-FILE tests
    print("\n=== Testing COMBINATIONS for each backbone ===")
    combo_auc_dict = {}  # {backbone: (mean_matrix, std_matrix)}
    combo_confusion_dict = {}
    combo_metrics_dict = {}

    for backbone in BACKBONES:
        print(f"\n========== Combination testing for backbone: {backbone} ==========")
        auc_matrix, std_matrix, confusion_matrices, metrics_list = test_combinations(backbone, FILE_IDS)
        combo_auc_dict[backbone] = (auc_matrix, std_matrix)
        combo_confusion_dict[backbone] = confusion_matrices
        combo_metrics_dict[backbone] = metrics_list

    # 3) Determine the global min AUC (we can fix the max at 1.0)
    global_min = find_global_auc_min(single_aucs_matrix, combo_auc_dict)
    global_max = 1.0
    print(f"\n[Info] Global AUC range across singles + combos: [{global_min:.3f}, {global_max:.3f}]")

    # 4) Plot single-file AUC heatmap
    print("\n=== Plotting SINGLE-FILE heatmap ===")
    plot_single_aucs_heatmap(
        single_aucs_matrix,
        single_aucs_stds_matrix,
        FILE_IDS,
        BACKBONES,
        OUTPUT_DIR,
        vmin=global_min,
        vmax=global_max
    )

    # 5) (Optional) Boxplots for each backbone across FOV=1..4
    print("\n=== Plotting BOX PLOTS + computing P-VALUES for each backbone (single-file) ===")
    for backbone in BACKBONES:
        plot_boxplot_and_compute_pvals(
            single_aucs_distribution,
            backbone,
            FILE_IDS,
            OUTPUT_DIR
        )

    # 6) Bar chart + significance across backbones at **FOV=2** 
    print("\n=== Comparing BACKBONES at FOV=2 ===")
    compare_backbones_for_fov(
        single_aucs_distribution,
        BACKBONES,
        fov=2,
        output_dir=OUTPUT_DIR
    )

    # 7) Plot combination-file heatmaps
    print("\n=== Plotting COMBINATION-FILE heatmaps ===")
    for backbone in BACKBONES:
        mean_mat, std_mat = combo_auc_dict[backbone]
        plot_triangle_heatmap_with_light_grey_diag(
            mean_mat,
            std_mat,
            FILE_IDS,
            backbone,
            OUTPUT_DIR,
            vmin=global_min,
            vmax=global_max
        )

        # Also print confusion matrices and metrics
        confusion_matrices = combo_confusion_dict[backbone]
        metrics_list = combo_metrics_dict[backbone]
        for (label1, label2, cm), (_, _, metrics) in zip(confusion_matrices, metrics_list):
            if cm is not None:
                #plot_confusion_matrix(cm, label1, label2, OUTPUT_DIR)
            print(f"\nMetrics for {label1} vs {label2}:")
            for mkey, mval in metrics.items():
                if pd.notnull(mval):
                    print(f"  {mkey}: {mval:.3f}")
                else:
                    print(f"  {mkey}: None")
