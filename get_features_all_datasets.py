import os
import numpy as np
import pandas as pd
import fnmatch

print('Starting...')

# Paths and parameters
meta_path = '/projects/rubinstein-lab/USERS/domans/Harmonized-metadata.csv'
meta_df = pd.read_csv(meta_path)
base_dirs = [
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-COG/results-cog-batch1-part1',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-COG/results-cog-batch1-part2',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-COG/results-cog-batch2',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-COG/results-cog-batch3',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-MGH/results-MGH-batch1',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-MGH/results-MGH-batch2',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-stjude/results-stjude',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-yale/results-yale-batch1',
    '/projects/rubinstein-lab/USERS/domans/pediSarcoma-yale/results-yale-batch2'
]
eccentricity_quantiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
area_deciles = eccentricity_quantiles
feature_deciles = list(range(5, 100, 10))

# List to store all feature data
all_features = []

# Iterate over Slide IDs in the metadata
for slide_id in meta_df['Slide ID']:
    valid_folder = None

    # Check for valid folder in all base directories
    for base_dir in base_dirs:
        # List all folders in the base directory
        potential_folders = [
            os.path.join(base_dir, folder)
            for folder in os.listdir(base_dir)
            if fnmatch.fnmatch(folder, f"{slide_id}*oid0")
        ]
        
        if potential_folders:
            # Select the first matching folder (assumes unique valid folders)
            valid_folder = potential_folders[0]
            break  # Stop checking other base directories if a valid folder is found

    # Skip if no valid folder was found
    if not valid_folder:
        print(f"Folder for Slide ID {slide_id} not found in the specified paths.")
        continue

    # Process the valid folder
    # Process 20x features
    features_20x = os.path.join(valid_folder, 'features', 'false-2-ctranspath_features.tsv.gz')
    if os.path.exists(features_20x):
        df_20x = pd.read_csv(features_20x, compression='gzip')
        feats_2 = df_20x.drop(columns=[
            'barcode', 'array_col', 'array_row', 'in_tissue',
            'pxl_row_in_fullres', 'pxl_col_in_fullres',
            'pxl_row_in_wsi', 'pxl_col_in_wsi'
        ])
        feats_2 = np.ravel(np.percentile(feats_2, feature_deciles, axis=0))
    else:
        print(f"20x features not found for {slide_id}")
        continue  # Skip processing if critical data is missing

    # Process 10x features
    features_10x = os.path.join(valid_folder, 'features', 'false-1-ctranspath_features.tsv.gz')
    if os.path.exists(features_10x):
        df_10x = pd.read_csv(features_10x, compression='gzip')
        feats_1 = df_10x.drop(columns=[
            'barcode', 'array_col', 'array_row', 'in_tissue',
            'pxl_row_in_fullres', 'pxl_col_in_fullres',
            'pxl_row_in_wsi', 'pxl_col_in_wsi'
        ])
        feats_1 = np.ravel(np.percentile(feats_1, feature_deciles, axis=0))
    else:
        print(f"10x features not found for {slide_id}")
        continue  # Skip processing if critical data is missing

    # Process per nucleus data
    csv_path = os.path.join(valid_folder, 'nucseg/per_nucleus_data.csv.gz')
    if os.path.exists(csv_path):
        df_nucleus = pd.read_csv(csv_path, compression='gzip')

        if 'eccentricity' in df_nucleus.columns:
            eccentricity_quantiles_values = np.percentile(df_nucleus['eccentricity'], eccentricity_quantiles).tolist()
        else:
            eccentricity_quantiles_values = [np.nan] * len(eccentricity_quantiles)

        if 'area' in df_nucleus.columns:
            area_deciles_values = np.percentile(df_nucleus['area'], area_deciles).tolist()
        else:
            area_deciles_values = [np.nan] * len(area_deciles)
    else:
        print(f"Nucleus data not found for {slide_id}")
        eccentricity_quantiles_values = [np.nan] * len(eccentricity_quantiles)
        area_deciles_values = [np.nan] * len(area_deciles)

    # Concatenate features
    feats_concat = np.concatenate((
        feats_2, feats_1,
        np.array(eccentricity_quantiles_values).flatten(),
        np.array(area_deciles_values).flatten()
    ))

    # Add subtype label
    histological_subtype = meta_df.loc[meta_df['Slide ID'] == slide_id, 'Histological Subtype'].values[0]
    if 'Ewing' in str(histological_subtype):
        feats_concat = np.append(feats_concat, 0)
    else:
        feats_concat = np.append(feats_concat, 1)

    # Add Slide ID as the final column
    feats_concat = np.append(feats_concat, slide_id)

    # Append to the features list
    all_features.append(feats_concat)

# Convert the list of arrays to a NumPy array
ctp_df_morpho_both = np.array(all_features, dtype=object)  # Use dtype=object for mixed types

# Optional: Save the result to a file or further process it
np.save('processed_features_morpho_both.npy', ctp_df_morpho_both)

print("Feature extraction, eccentricity quantile, area decile calculation, and Slide ID addition completed.")
