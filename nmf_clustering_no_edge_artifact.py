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


# Define the directory containing the core images
core_dir = 'output'

# Set the tile size and number of clusters
tile_size = (30, 30)  # Adjust as needed
n_clusters = 12  # Set this to 8 based on your clustering needs

# Store all tiles and their positions (across all cores)
all_tiles = []
all_positions = []
core_identifiers = []

# Selected proteins for faster NMF clustering (excluding DAPI channel)
selected_proteins = ['Ki67', 'aSMA', 'CD11c', 'Vimentin', 'CD45', 'CD31', 'p21', 'CD4', 'CD19']
# Example: If these proteins correspond to specific channel indices in your data, you can map them here
channel_indices = [1, 2, 6, 7, 8, 11, 15, 18, 20]  # Replace with correct indices for selected proteins

# Function to split image into tiles, using the combined signal from all selected channels to skip background tiles
def split_into_tiles_with_signal_check(full_img_np, reduced_img_np, tile_size, expected_channels, signal_threshold=1):
    tiles = []
    positions = []
    tile_mask = []  # To keep track of valid (high signal) and invalid (low signal) tiles
    num_channels, img_h, img_w = reduced_img_np.shape  # First dimension is number of channels, then height, then width
    
    # Iterate over the image to create tiles
    for i in range(0, img_h, tile_size[0]):
        for j in range(0, img_w, tile_size[1]):
            tile = reduced_img_np[:, i:i+tile_size[0], j:j+tile_size[1]]  # Extract tile across selected channels
            full_tile = full_img_np[:, i:i+tile_size[0], j:j+tile_size[1]]  # Full tile including all channels

            # Check the combined signal from all channels
            combined_signal = np.mean(full_tile, axis=(1, 2))  # Average signal for each channel across the tile
            total_signal = np.sum(combined_signal)  # Sum of all channel signals for the tile

            # Skip tiles with low combined signal
            if total_signal < signal_threshold:
                tile_mask.append(-1)  # Use -1 for low-signal tiles (to be displayed as white)
                continue
            
            # Check if tile size matches the desired tile size, pad if necessary
            if tile.shape[1] != tile_size[0] or tile.shape[2] != tile_size[1]:
                pad_h = tile_size[0] - tile.shape[1]
                pad_w = tile_size[1] - tile.shape[2]
                tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            
            # Aggregate the tile by computing the mean intensity for each marker (channel)
            mean_tile = np.mean(tile, axis=(1, 2))  # Average across height and width, keep channel dimension
            tiles.append(mean_tile)  # Store the mean intensities for the markers
            positions.append((i, j))  # Store the top-left corner of the tile
            tile_mask.append(1)  # Mark this tile as valid (high signal)
    
    return tiles, positions, tile_mask


# Loop through all core images in the directory that start with 'reg00' but exclude those starting with 'reg007' or 'reg0010'
tile_masks = []
for core_filename in os.listdir(core_dir):
    if core_filename.startswith('reg00') and not (core_filename.startswith('reg007') or core_filename.startswith('reg0010')) and (core_filename.endswith('.tif') or core_filename.endswith('.qptiff')):
        # Load the core image (including all channels, e.g., DAPI and proteins)
        file_path = os.path.join(core_dir, core_filename)
        with tifffile.TiffFile(file_path) as tif:
            core_image = tif.asarray()  # Load image data as numpy array
            
            # Subset the image to the selected proteins/channels (excluding DAPI)
            reduced_core_image = core_image[channel_indices, :, :]  # Select only the chosen protein channels
            
            # Split the core image into tiles, using the DAPI signal from the full image to skip background tiles
            dapi_signal_threshold = 6  # Adjust this threshold based on your data
            tiles, positions, tile_mask = split_into_tiles_with_signal_check(core_image, reduced_core_image, tile_size, len(selected_proteins), signal_threshold=dapi_signal_threshold)

        
        # Store tiles, positions, core identifiers, and masks
        all_tiles.extend(tiles)
        all_positions.extend(positions)
        core_identifiers.extend([core_filename] * len(tiles))  # Track which core the tile comes from
        tile_masks.extend(tile_mask)  # Track the mask for each tile (whether it's background or not)

# Convert all tiles to a 2D array for clustering
all_tiles_array = np.vstack(all_tiles)  # No need to flatten further since tiles are already reduced to 9 values per channel

# Normalize the data using Z-Score normalization (Standardization)
scaler = StandardScaler()
all_tiles_array_normalized = scaler.fit_transform(all_tiles_array)  # Standardize to have mean=0 and std=1
min_value = np.min(all_tiles_array_normalized)
if min_value < 0:
    all_tiles_array_normalized += abs(min_value)
print("Finished normalizing")

# Perform NMF clustering on the normalized tiles
nmf = NMF(n_components=n_clusters, init='random', max_iter=100)  # Use 'random' initialization for faster convergence
W = nmf.fit_transform(all_tiles_array_normalized)  # W contains the cluster labels for each tile
H = nmf.components_

# Save the cluster labels for each tile
cluster_labels = np.argmax(W, axis=1)  # Get the dominant cluster for each tile

# Add the tile mask to the cluster labels (mark low-DAPI tiles as -1)
#cluster_labels = np.array([label if tile_masks[idx] != -1 else -1 for idx, label in enumerate(cluster_labels)])

# Save the cluster labels and their corresponding core and position
output_data = []
for idx, label in enumerate(cluster_labels):
    core_file = core_identifiers[idx]
    position = all_positions[idx]
    output_data.append((core_file, position, label))

# Save the output data to a file (e.g., CSV or JSON)
output_csv = 'tile_cluster_labels.csv'
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Core_File', 'Tile_Position', 'Cluster_Label'])
    writer.writerows(output_data)

print(f"Cluster labels saved to {output_csv}")

# Replot tiles with cluster labels and add a legend
def plot_clustered_core_without_padding(core_filename):
    core_data = [row for row in output_data if row[0] == core_filename]
    
    # Reload the core image
    file_path = os.path.join(core_dir, core_filename)
    with tifffile.TiffFile(file_path) as tif:
        core_image = tif.asarray()  # Reload core image
    
    # Create an empty array to store the clustered image, initialized to -1 (to be plotted as white)
    clustered_image = np.full((core_image.shape[1], core_image.shape[2]), -1)  # Use -1 for background tiles
    
    # Place the clustered tiles back in their original positions
    for core_file, position, label in core_data:
        i, j = position  # Get the top-left corner of the tile
        # Skip tiles that don't match the required size
        if i + tile_size[0] > core_image.shape[1] or j + tile_size[1] > core_image.shape[2]:
            continue  # Skip tiles that would exceed the core image dimensions
        if label != -1:  # Only assign a label if the tile is not low-DAPI
            clustered_image[i:i+tile_size[0], j:j+tile_size[1]] = label
    
    # Create a custom colormap where -1 is white, and the rest are normal cluster colors
    cmap = plt.get_cmap('jet')  # Get the 'jet' colormap
    new_colors = cmap(np.linspace(0, 1, n_clusters))  # Generate n_clusters colors from 'jet'
    white = np.array([1, 1, 1, 1])  # White for background
    new_colors = np.vstack([white, new_colors])  # Add white color for -1 (background)
    custom_cmap = ListedColormap(new_colors)
    
    # Plot the clustered core without interpolation
    plt.figure(figsize=(10, 10))
    plt.imshow(clustered_image, cmap=custom_cmap, vmin=-1, vmax=n_clusters-1, interpolation='nearest')  # Disable interpolation
    plt.title(f'Clustered Core: {core_filename}')
    
    # Adjust the legend to exclude the white background (-1) and show only n_clusters
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude background (-1)
    patches = [mpatches.Patch(color=new_colors[i + 1], label=f'Cluster {i}') for i in range(n_clusters)]  # Offset by 1 to skip white
    
    # Add the legend to the plot
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
plot_clustered_core_without_padding('reg009_X01_Y01_Z01.tif')

# Assuming H is the matrix from NMF (nmf.components_)
# selected_proteins contains the names of the markers used

# Get the number of clusters and markers
n_clusters, n_markers = H.shape

# For each cluster, find the top contributing markers
top_markers_per_cluster = 5  # Number of top markers to display per cluster (adjust as needed)

for cluster_idx in range(n_clusters):
    print(f"\nCluster {cluster_idx}:")
    
    # Get the contributions of each marker to the current cluster
    cluster_contributions = H[cluster_idx]
    
    # Get the indices of the top markers
    top_marker_indices = np.argsort(cluster_contributions)[::-1][:top_markers_per_cluster]  # Sort descending
    
    # Print the top markers and their contribution scores
    for marker_idx in top_marker_indices:
        marker_name = selected_proteins[marker_idx]  # Get the marker name
        contribution_score = cluster_contributions[marker_idx]  # Get the contribution score
        print(f"{marker_name}: {contribution_score:.4f}")
