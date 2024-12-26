import random
import matplotlib.pyplot as plt

# Set the number of random tiles to display
num_random_tiles = 10

# Randomly select 10 tile indices
random_tile_indices = random.sample(range(len(W)), num_random_tiles)

# Plot the component contributions for each randomly selected tile
for tile_idx in random_tile_indices:
    components = range(n_clusters)
    contributions = W[tile_idx]
    
    plt.figure(figsize=(6, 4))
    plt.bar(components, contributions)
    plt.xlabel('NMF Component')
    plt.ylabel('Contribution')
    plt.title(f'Component Contributions for Tile {tile_idx}')
    plt.show()
