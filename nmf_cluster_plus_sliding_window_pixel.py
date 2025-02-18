import os
import numpy as np
import tifffile
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import cm

##############################################################################
#                                PARAMETERS
##############################################################################

# Directory containing cores
core_dir = 'output'

# Tile settings
tile_size = (30, 30)  # (height, width) of each tile
step = (15, 15)       # how far to move in each step (smaller => more overlap)

# NMF cluster settings
n_clusters = 12

# Markers to include in NMF (excluding DAPI, etc.)
selected_proteins = ['Ki67', 'aSMA', 'CD11c', 'Vimentin', 'CD45', 'CD31', 'p21', 'CD4', 'CD19']
# Map these proteins to the correct channel indices in your data
channel_indices = [1, 2, 6, 7, 8, 11, 15, 18, 20]  

# Threshold to skip low‐signal tiles (based on DAPI or combined signal)
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
    full_img_np:   Full image array of shape (channels, height, width) 
                   (including DAPI and all other channels).
    reduced_img_np: Subset of channels for NMF (already indexed).
    tile_size:     (tile_h, tile_w)
    step:          (step_h, step_w) - can be smaller than tile_size for overlap.
    signal_threshold: Skip tiles whose sum of average signal across 
                      channels < signal_threshold.
    
    Returns:
        tiles:      list of mean intensity vectors (one per tile).
        positions:  list of (i, j) top-left corner for each tile.
        tile_mask:  list of 1 or -1 (1 = keep tile, -1 = skip).
    """
    tiles = []
    positions = []
    tile_mask = []

    num_channels, img_h, img_w = reduced_img_np.shape
    tile_h, tile_w = tile_size
    step_h, step_w = step

    # Sliding window with possible overlap
    for i in range(0, img_h - tile_h + 1, step_h):
        for j in range(0, img_w - tile_w + 1, step_w):
            # Extract the tile for selected channels
            tile = reduced_img_np[:, i:i+tile_h, j:j+tile_w]
            # Extract the tile for ALL channels (to compute DAPI or total signal)
            full_tile = full_img_np[:, i:i+tile_h, j:j+tile_w]

            # Check combined signal across all channels in 'full_tile'
            combined_signal = np.mean(full_tile, axis=(1, 2))  # average of each channel
            total_signal = np.sum(combined_signal)

            # Skip tiles with low combined signal
            if total_signal < signal_threshold:
                tile_mask.append(-1)
                continue

            # Compute mean intensity for each selected channel
            mean_tile = np.mean(tile, axis=(1, 2))  # shape: (num_selected_channels,)
            tiles.append(mean_tile)
            positions.append((i, j))
            tile_mask.append(1)

    return tiles, positions, tile_mask


##############################################################################
#          FUNCTION TO BUILD A PIXEL-LEVEL MEMBERSHIP MAP (AVERAGE)
##############################################################################

def create_pixel_level_membership_map(W, positions, tile_size, img_shape):
    """
    Create a pixel-level membership map by averaging the cluster membership
    vectors from all tiles that cover each pixel.
    
    Args:
        W:          (num_tiles, n_clusters) - the NMF membership matrix (W).
        positions:  list of (top_left_row, top_left_col) for each tile.
        tile_size:  (tile_h, tile_w)
        img_shape:  (height, width) of the full image
        
    Returns:
        membership_map: shape (height, width, n_clusters) with average membership
                        for each pixel.
    """
    tile_h, tile_w = tile_size
    H, W_img = img_shape
    n_clusters = W.shape[1]

    # We'll accumulate membership sums, then divide by overlap count.
    membership_sum = np.zeros((H, W_img, n_clusters), dtype=np.float32)
    overlap_count = np.zeros((H, W_img), dtype=np.float32)

    for tile_idx, (row, col) in enumerate(positions):
        # The cluster membership vector for this tile:
        tile_membership = W[tile_idx, :]  # shape: (n_clusters,)

        # Add it to all pixels in [row:row+tile_h, col:col+tile_w]
        membership_sum[row:row+tile_h, col:col+tile_w, :] += tile_membership
        overlap_count[row:row+tile_h, col:col+tile_w] += 1

    # Average membership
    eps = 1e-7
    membership_map = membership_sum / (overlap_count[..., None] + eps)

    return membership_map


##############################################################################
#                               MAIN SCRIPT
##############################################################################

# Store all tiles (from all cores) for global NMF
all_tiles = []
all_positions = []
core_identifiers = []
tile_masks = []

# Loop over all cores in the directory
for core_filename in os.listdir(core_dir):
    # Example condition: keep files that start with 'reg00' but exclude reg007, reg0010
    if (core_filename.startswith('reg00') 
        and not (core_filename.startswith('reg007') or core_filename.startswith('reg0010'))
        and (core_filename.endswith('.tif') or core_filename.endswith('.qptiff'))):

        file_path = os.path.join(core_dir, core_filename)
        with tifffile.TiffFile(file_path) as tif:
            core_image = tif.asarray()  # shape: (channels, height, width)

        # Subset the image to selected_proteins
        reduced_core_image = core_image[channel_indices, :, :]

        # Create overlapping tiles
        tiles, positions, tile_mask = sliding_window_tiles_with_signal_check(
            full_img_np=core_image,
            reduced_img_np=reduced_core_image,
            tile_size=tile_size,
            step=step,
            signal_threshold=signal_threshold
        )

        # Append to global lists
        all_tiles.extend(tiles)
        all_positions.extend([(core_filename, pos) for pos in positions]) 
        # (store which core the tile came from along with tile position)
        tile_masks.extend(tile_mask)
        # We'll store core_identifiers in parallel with tiles
        # but we can do it more explicitly with (core_filename, pos) above
        # or simply:
        # core_identifiers.extend([core_filename]*len(tiles))

# Convert list of tile vectors to 2D array: (num_tiles, num_markers)
all_tiles_array = np.vstack(all_tiles)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
all_tiles_array_normalized = scaler.fit_transform(all_tiles_array)


# Perform NMF
nmf = NMF(n_components=n_clusters, init='random', max_iter=100, random_state=0)
W = nmf.fit_transform(all_tiles_array_normalized)  # shape: (num_tiles, n_clusters)
H = nmf.components_

# The "dominant" cluster label at tile-level
tile_cluster_labels = np.argmax(W, axis=1)

# Save tile-level results to CSV
output_csv = 'tile_cluster_labels.csv'
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Core_File', 'Tile_Position', 'Cluster_Label'])
    for idx, label in enumerate(tile_cluster_labels):
        core_file, pos = all_positions[idx]
        writer.writerow([core_file, pos, label])

print(f"Tile-level cluster labels saved to {output_csv}")

##############################################################################
#                   OPTIONAL: PER-CORE PIXEL-LEVEL MAPS
##############################################################################
# We can generate a pixel-level cluster map for each core separately. 
# Because we combined all tiles for a single NMF, we must:
#    1. Identify which tiles belong to that core.
#    2. Gather W for those tiles.
#    3. Reproject to pixel level (averaging membership).
#    4. Argmax across clusters to get final label per pixel.

def plot_pixel_level_clusters(core_filename):
    """Create and display a pixel-level cluster map for a given core."""
    # Load core image again for shape
    file_path = os.path.join(core_dir, core_filename)
    with tifffile.TiffFile(file_path) as tif:
        core_image = tif.asarray()
    core_height, core_width = core_image.shape[1], core_image.shape[2]

    # Extract the tile indices for this core
    core_tile_indices = [i for i, (cfile, _) in enumerate(all_positions) if cfile == core_filename]
    if len(core_tile_indices) == 0:
        print(f"No tiles found for core {core_filename} (maybe threshold skipped them).")
        return

    # Gather membership vectors and positions
    W_core = W[core_tile_indices, :]  # shape: (#tiles_for_this_core, n_clusters)
    core_positions = [all_positions[i][1] for i in core_tile_indices]  # the (row, col) positions

    # Build the pixel-level membership map
    membership_map = create_pixel_level_membership_map(
        W_core, 
        core_positions, 
        tile_size=tile_size, 
        img_shape=(core_height, core_width)
    )  # shape: (H, W, n_clusters)

    # Argmax for cluster labels
    pixel_labels = np.argmax(membership_map, axis=2)

    # Create colormap: let's say cluster labels range 0..n_clusters-1
    # We'll build a "jet" colormap with n_clusters distinct colors
    cmap = plt.get_cmap('jet', n_clusters)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(pixel_labels, cmap=cmap, vmin=0, vmax=n_clusters-1, interpolation='nearest')
    plt.title(f'Pixel-Level Clusters (smoothed) for {core_filename}')

    # Build legend
    patches = []
    for c in range(n_clusters):
        patches.append(mpatches.Patch(color=cmap(c), label=f'Cluster {c}'))
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

##############################################################################
#  EXAMPLE: Plot one of the core images at pixel-level
##############################################################################

# Replace 'reg009_X01_Y01_Z01.tif' with a valid filename from your data
core_to_plot = 'reg009_X01_Y01_Z01.tif'
plot_pixel_level_clusters(core_to_plot)

##############################################################################
#     PRINT TOP CONTRIBUTING MARKERS PER CLUSTER (From the original script)
##############################################################################

n_clusters_H, n_markers = H.shape
top_markers_per_cluster = 5

for cluster_idx in range(n_clusters_H):
    print(f"\nCluster {cluster_idx}:")
    cluster_contributions = H[cluster_idx]
    # Sort descending
    top_marker_indices = np.argsort(cluster_contributions)[::-1][:top_markers_per_cluster]
    for marker_idx in top_marker_indices:
        marker_name = selected_proteins[marker_idx]  # map index -> protein name
        contribution_score = cluster_contributions[marker_idx]
        print(f"{marker_name}: {contribution_score:.4f}")
