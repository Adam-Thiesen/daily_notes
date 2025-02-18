import anndata as ad
import pandas as pd
import plotly.graph_objects as go
import random

# Load the .h5ad file
adata = ad.read_h5ad('/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.uni-2.h5ad')

# Define the columns to use
columns_to_use = ['Contributor', 'Age bin', 'Histological Type', 'Histological Subtype', 'Race', 'Sex']
obs_df = adata.obs[columns_to_use].copy()

# Make each 'NA' unique per column to prevent merging across different categories
for col in columns_to_use:
    obs_df[col] = obs_df[col].astype(str)
    obs_df[col] = obs_df[col].replace(['NA', 'nan', 'None'], f'NA ({col})')

# Create a list of all unique nodes
unique_nodes = pd.unique(obs_df.values.ravel())
node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

# Generate Sankey plot links
sankey_data = []
for i in range(len(columns_to_use) - 1):
    source_col = columns_to_use[i]
    target_col = columns_to_use[i + 1]
    link_counts = obs_df.groupby([source_col, target_col]).size().reset_index(name='count')
    
    for _, row in link_counts.iterrows():
        source = row[source_col]
        target = row[target_col]
        count = row['count']
        sankey_data.append({
            'source': node_mapping[source],
            'target': node_mapping[target],
            'value': count
        })

# Prepare data for Sankey plot
sources = [d['source'] for d in sankey_data]
targets = [d['target'] for d in sankey_data]
values = [d['value'] for d in sankey_data]

# Generate **brighter, happier colors** for each node
def bright_color():
    """Generate a bright and vibrant random color."""
    return f'rgba({random.randint(150, 255)}, {random.randint(100, 255)}, {random.randint(150, 255)}, 0.85)'

node_colors = [bright_color() for _ in unique_nodes]

# Create Sankey plot
fig = go.Figure(go.Sankey(
    node=dict(
        pad=30,  # Increase padding between nodes
        thickness=30,  # Increase node thickness
        line=dict(color="black", width=1),  # Add borders to nodes
        label=list(node_mapping.keys()),  # Node labels
        color=node_colors  # Assign bright colors per node
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color="rgba(120, 120, 120, 0.2)"  # Softer link color for better contrast
    )
))

# Adjust font size and title
fig.update_layout(
    title_text="Sankey Diagram with Brighter Colors 🎨",
    font=dict(size=16, family="Arial Black, sans-serif", color="black"),  # Set text size & bold
    height=900,  # Increase height for better spacing
    width=1400  # Increase width for better layout
)

# Save as HTML
fig.write_html("sankey_plot_bright.html")

# Show the plot
fig.show()
