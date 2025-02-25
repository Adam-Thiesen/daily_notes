import os
import numpy as np
import tifffile
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


##############################################################################
#                                PARAMETERS
##############################################################################

core_dir = 'output'       # Directory containing your images
tile_size = (50, 50)      # (height, width) of each tile
step = (15, 15)           # Overlap step (smaller than tile_size => overlapping)
n_clusters = 12           # Number of NMF clusters

# Channels / Markers
selected_proteins = ['Ki67', 'aSMA', 'CD11c', 'Vimentin', 'CD45', 'CD31', 'p21', 'CD4', 'CD19']
channel_indices = [1, 2, 6, 7, 8, 11, 15, 18, 20]

# Threshold to skip low-signal tiles
signal_threshold = 6

##############################################################################
#                    SLIDING-WINDOW TILING FUNCTION
##############################################################################

def sliding_window_tiles_with_signal_check(
    full_img_np,
    reduced_img_np,
    tile_size,
    step,
    signal_threshold=1
):
    """
    Splits an image into overlapping tiles, skipping tiles with low total signal.
    
    Args:
        full_img_np:       (channels, height, width), includes all channels (DAPI, etc.)
        reduced_img_np:    subset of channels used for NMF
        tile_size:         (tile_h, tile_w)
        step:              (step_h, step_w) - overlap step
        signal_threshold:  skip tiles if sum of mean intensities < this value

    Returns:
        tiles:       list of mean intensity vectors (num_selected_channels,)
        positions:   list of (row, col) top-left corners
        tile_mask:   list of 1 or -1 (kept vs. skipped)
    """
    tiles = []
    positions = []
    tile_mask = []

    num_channels, img_h, img_w = reduced_img_np.shape
    tile_h, tile_w = tile_size
    step_h, step_w = step

    for i in range(0, img_h - tile_h + 1, step_h):
        for j in range(0, img_w - tile_w + 1, step_w):
            tile = reduced_img_np[:, i:i+tile_h, j:j+tile_w]
            full_tile = full_img_np[:, i:i+tile_h, j:j+tile_w]

            combined_signal = np.mean(full_tile, axis=(1, 2))
            total_signal = np.sum(combined_signal)

            # Skip low-signal tiles
            if total_signal < signal_threshold:
                tile_mask.append(-1)
                continue

            # Mean intensities for the selected markers
            mean_tile = np.mean(tile, axis=(1, 2))
            tiles.append(mean_tile)
            positions.append((i, j))
            tile_mask.append(1)

    return tiles, positions, tile_mask

##############################################################################
#  CREATE A PIXEL-LEVEL MEMBERSHIP MAP (AVERAGING TILE MEMBERSHIPS + OVERLAP)
##############################################################################

def create_pixel_level_membership_map(W, positions, tile_size, img_shape):
    """
    Build a pixel-level membership map by averaging each tile's cluster membership
    over all pixels it covers. Also return overlap_count to identify pixels not covered.
    
    Args:
        W:          (num_tiles, n_clusters) - NMF membership for each tile
        positions:  list of (top_left_row, top_left_col) for each tile
        tile_size:  (tile_h, tile_w)
        img_shape:  (height, width) of the full image

    Returns:
        membership_map: (height, width, n_clusters)
        overlap_count:  (height, width) - how many tiles covered each pixel
    """
    tile_h, tile_w = tile_size
    H, W_img = img_shape
    n_clusters = W.shape[1]

    membership_sum = np.zeros((H, W_img, n_clusters), dtype=np.float32)
    overlap_count = np.zeros((H, W_img), dtype=np.float32)

    # Accumulate memberships for each overlapping tile
    for tile_idx, (row, col) in enumerate(positions):
        tile_membership = W[tile_idx, :]  # shape: (n_clusters,)
        membership_sum[row:row+tile_h, col:col+tile_w, :] += tile_membership
        overlap_count[row:row+tile_h, col:col+tile_w] += 1

    # Average membership
    eps = 1e-7
    membership_map = membership_sum / (overlap_count[..., None] + eps)

    return membership_map, overlap_count

##############################################################################
#                      GATHER ALL TILES FOR GLOBAL NMF
##############################################################################

all_tiles = []
all_positions = []  # list of (core_filename, (row, col))
tile_masks = []

for core_filename in os.listdir(core_dir):
    # Example: only process files starting with 'reg00' except reg007 & reg0010
    if (core_filename.startswith('reg00') 
        and not (core_filename.startswith('reg007') or core_filename.startswith('reg0010'))
        and (core_filename.endswith('.tif') or core_filename.endswith('.qptiff'))):

        file_path = os.path.join(core_dir, core_filename)
        with tifffile.TiffFile(file_path) as tif:
            core_image = tif.asarray()  # (channels, height, width)

        # Subset to selected proteins
        reduced_core_image = core_image[channel_indices, :, :]

        # Overlapping tiles
        tiles, positions, tile_mask = sliding_window_tiles_with_signal_check(
            full_img_np=core_image,
            reduced_img_np=reduced_core_image,
            tile_size=tile_size,
            step=step,
            signal_threshold=signal_threshold
        )

        # Append
        all_tiles.extend(tiles)
        # store (core_filename, (i,j)) for each tile
        all_positions.extend([(core_filename, pos) for pos in positions])
        tile_masks.extend(tile_mask)

# Convert tile list to a numpy array
all_tiles_array = np.vstack(all_tiles)  # shape: (num_tiles, num_selected_channels)

##############################################################################
#                      SCALING & NMF
##############################################################################

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
all_tiles_array_normalized = scaler.fit_transform(all_tiles_array)

nmf = NMF(n_components=n_clusters, init='random', max_iter=100, random_state=0)
W = nmf.fit_transform(all_tiles_array_normalized)  # (num_tiles, n_clusters)
H = nmf.components_

# Tile-level cluster labels (argmax across membership vector)
tile_cluster_labels = np.argmax(W, axis=1)

# Save tile-level results
output_csv = 'tile_cluster_labels.csv'
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Core_File', 'Tile_Position', 'Cluster_Label'])
    for idx, label in enumerate(tile_cluster_labels):
        core_file, pos = all_positions[idx]
        writer.writerow([core_file, pos, label])

print(f"Tile-level cluster labels saved to {output_csv}")

##############################################################################
#     PLOT PER-CORE PIXEL-LEVEL MAPS (WITH BACKGROUND = -1)
##############################################################################

def plot_pixel_level_clusters(core_filename):
    """Create and display a pixel-level cluster map for the given core, 
       labeling uncovered pixels as -1 (background)."""
    
    # Load core image for shape
    file_path = os.path.join(core_dir, core_filename)
    with tifffile.TiffFile(file_path) as tif:
        core_image = tif.asarray()
    core_height, core_width = core_image.shape[1], core_image.shape[2]

    # Find relevant tiles
    core_tile_indices = [i for i,(cfile,_) in enumerate(all_positions) 
                         if cfile == core_filename]
    if len(core_tile_indices) == 0:
        print(f"No tiles found for core {core_filename}.")
        return

    # Gather the membership rows for just this core
    W_core = W[core_tile_indices, :]  # (#tiles_for_core, n_clusters)
    core_positions = [all_positions[i][1] for i in core_tile_indices]  # (row,col)

    # Build pixel-level membership map
    membership_map, overlap_count = create_pixel_level_membership_map(
        W_core, 
        core_positions, 
        tile_size, 
        (core_height, core_width)
    )  # membership_map.shape: (H, W, n_clusters)

    # Argmax across clusters
    pixel_labels = np.argmax(membership_map, axis=2)  # shape: (H, W)

    # Mark any pixels with no coverage as background (-1)
    pixel_labels[overlap_count < 1] = -1

    ############## Create a custom colormap ##############
    # We'll define -1 as white, then cluster 0..(n_clusters-1) as the "jet" palette.

    # Let's build a colormap array:
    from matplotlib import colors
    from matplotlib import cm
    import matplotlib.patches as mpatches

    # We'll get 'jet' with exactly n_clusters discrete colors
    jet_cmap = plt.get_cmap('jet', n_clusters)  
    # Extract RGBA array of shape (n_clusters,4)
    jet_colors = jet_cmap(range(n_clusters))
    
    # Insert white at index 0 for background
    white = np.array([1,1,1,1])  # RGBA
    new_colors = np.vstack([white, jet_colors])  # shape => (n_clusters+1, 4)
    
    custom_cmap = colors.ListedColormap(new_colors)
    
    # Because we placed background = -1, let's shift the data up by +1 
    # so that -1 maps to 0 (white), cluster 0 maps to 1, etc.
    shifted_labels = pixel_labels + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(shifted_labels, cmap=custom_cmap, 
               vmin=0, vmax=n_clusters, interpolation='nearest')
    plt.title(f'Pixel-Level Clusters (smoothed) for {core_filename}')

    # Build a legend. We'll skip the background in the legend or label it "BG".
    patches = []
    # background
    patches.append(mpatches.Patch(color=white, label='Background'))

    # clusters 0..n_clusters-1
    for c in range(n_clusters):
        # The color for cluster c is new_colors[c+1] (since we shifted by 1)
        color = new_colors[c+1]
        label_str = f'Cluster {c}'
        patch = mpatches.Patch(color=color, label=label_str)
        patches.append(patch)

    plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.show()

##############################################################################
# EXAMPLE USAGE: Plot one core
##############################################################################

core_to_plot = 'reg009_X01_Y01_Z01.tif'
plot_pixel_level_clusters(core_to_plot)

##############################################################################
#                PRINT TOP CONTRIBUTING MARKERS PER CLUSTER
##############################################################################

n_clusters_H, n_markers = H.shape
top_markers_per_cluster = 5

print("\nTop Markers per Cluster:")
for cluster_idx in range(n_clusters_H):
    print(f"\nCluster {cluster_idx}:")
    cluster_contributions = H[cluster_idx]
    # sort descending
    top_indices = np.argsort(cluster_contributions)[::-1][:top_markers_per_cluster]
    for midx in top_indices:
        marker_name = selected_proteins[midx]
        contrib_score = cluster_contributions[midx]
        print(f"  {marker_name}: {contrib_score:.4f}")
