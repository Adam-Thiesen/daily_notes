import os
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import math
from collections import defaultdict

class RhabdomyosarcomaDataset(Dataset):
    def __init__(self, data_dir, labels_df, start_col=8, num_features=768):
        self.data = []
        self.labels = []
        self.image_ids = []
        self.tile_indices = []
        self.labels_df = labels_df
        self.data_dir = data_dir
        self.start_col = start_col
        self.num_features = num_features
        
        self.labels_df['slide_id'] = self.labels_df['slide_id'].astype(str).str.strip()
        
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.gz'):
                base_name = file_name.replace('.gz', '').strip()
                label_row = self.labels_df[self.labels_df['slide_id'] == base_name]
                if label_row.empty:
                    print(f"Warning: No label found for file {file_name}, skipping...")
                    continue

                file_path = os.path.join(data_dir, file_name)
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f)
                    features = df.iloc[:, self.start_col:self.start_col + self.num_features].apply(pd.to_numeric, errors='coerce').fillna(0).values
                    self.data.append(features)
                    self.tile_indices.append(df.index.to_list())
                    label = int(label_row['labels'].values[0])
                    self.labels.append(label)
                    self.image_ids.append(base_name)
        
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long), self.image_ids[idx], self.tile_indices[idx]

class SimpleTransformerWithFeatureAttention(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(SimpleTransformerWithFeatureAttention, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x):
        batch_size, num_tiles, _ = x.size()
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Switch batch and sequence dimensions for transformer
        x = self.transformer(x)  # Self-attention within the tile
        
        self.attention_weights = x  # Store attention weights before aggregation
        
        x = x.mean(dim=0)  # Aggregate tile embeddings
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1), self.attention_weights.transpose(0, 1)  # Return attention weights per tile

# SaveOutput class to capture attention weights
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])  # Append attention weights

    def clear(self):
        self.outputs = []

# Function to patch the attention layer
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    m.forward = wrap

# Load the labels
labels_df = pd.read_csv('filtered_labels.csv')

# Map string labels to integers
label_mapping = {'ALVEOLAR': 0, 'EMBRYONAL': 1}
labels_df['labels'] = labels_df['labels'].map(label_mapping)
print(labels_df.head())

# Initialize the dataset
data_dir = '/flashscratch/thiesa/img_features'
dataset = RhabdomyosarcomaDataset(data_dir, labels_df)

# Create a unique list of image IDs and their corresponding labels
image_ids = np.array(dataset.image_ids)
unique_image_ids = np.unique(image_ids)
unique_labels = [dataset.labels[np.where(image_ids == img_id)[0][0]] for img_id in unique_image_ids]

# Perform the train-test split on the unique image IDs
train_image_ids, test_image_ids, train_labels, test_labels = train_test_split(
    unique_image_ids, unique_labels, test_size=0.25, stratify=unique_labels, random_state=42
)

# Use the split image IDs to create the train and test datasets
train_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in train_image_ids]
test_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in test_image_ids]

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Create DataLoaders without collate function, using batch_size=1
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_counts = np.bincount(dataset.labels)
majority_class_count = max(class_counts)
minority_class_count = min(class_counts)

# Calculate pos_weight as the ratio of majority to minority
pos_weight_value = majority_class_count / minority_class_count
print(pos_weight_value)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

# Initialize model, loss function, and optimizer
model = SimpleTransformerWithFeatureAttention(input_dim=768)
model.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(model, dataloader, epochs=75):
    model.train()
    epoch_losses = []  # List to store loss at each epoch
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels, image_ids, tile_indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.view(-1)  # Ensure labels have shape [batch_size]
            
            optimizer.zero_grad()
            outputs, attn_weights = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')
    
    # Save the model after training
    torch.save(model.state_dict(), 'model.pth')

    # Plot the loss over epochs
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.show()

# Train the model
train(model, train_loader, epochs=75)

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

# Assuming 'model' is your existing model and you have at least 2 GPUs available.
model = SimpleTransformerWithFeatureAttention(input_dim=768)
model.load_state_dict(torch.load('model.pth'))

patch_attention(model.transformer.layers[-1].self_attn)
save_output = SaveOutput()
hook_handle = model.transformer.layers[-1].self_attn.register_forward_hook(save_output)

model.to('cuda')
model.eval()

def evaluate_single_image(model, dataset, index):
    model.eval()
    attention_weights_dict = {}
    correct_images = []
    incorrect_images = []

    with torch.no_grad():
        # Get the specific batch
        inputs, labels, image_ids, tile_indices = dataset[index]
        inputs, labels = inputs.unsqueeze(0).to(device), torch.tensor([labels]).to(device).float()
        labels = labels.view(-1)  # Ensure labels have shape [batch_size]
        
        outputs, attn_weights = model(inputs)
        probs = torch.sigmoid(outputs)  # Apply sigmoid for probabilities
        
        img_id = image_ids
        print(f'Image ID: {img_id}')
        
        # Reduce the attention weights to a single value per tile
        reduced_attention_weights = attn_weights[0].mean(dim=1).cpu().numpy()
        reduced_attention_weights = torch.softmax(torch.tensor(reduced_attention_weights), dim=0).numpy()
        attention_weights_dict[img_id] = reduced_attention_weights

        # Determine if the image was correctly classified
        pred_label = (probs[0] > 0.5).float().item()
        print(f'Predicted Label: {pred_label}')
        true_label = labels[0].item()
        if pred_label == true_label:
            correct_images.append(img_id)
        else:
            incorrect_images.append(img_id)

        torch.cuda.empty_cache()

    return pred_label, true_label, correct_images, incorrect_images, attention_weights_dict

# Assuming `model`, `test_loader`, and `device` are already defined
# Extract the dataset from the DataLoader
dataset = test_loader.dataset

index = 0  # Change this to evaluate different indices
pred_label, true_label, correct_images, incorrect_images, attention_weights = evaluate_single_image(model, dataset, index)

print(f'True Label: {true_label}')
print(f'Attention Weights: {attention_weights}')
print(f'Correct Images: {correct_images}')
print(f'Incorrect Images: {incorrect_images}')

import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Function to extract tile positions and features from a given .gz file
def extract_tile_positions_and_features(file_path, start_col=8, num_features=768):
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f)
        array_row = df['array_row'].values
        array_col = df['array_col'].values
        features = df.iloc[:, start_col:start_col + num_features].apply(pd.to_numeric, errors='coerce').fillna(0).values
    return array_row, array_col, features

# Function to plot the attention heatmap for a given image in the test set
def plot_attention_heatmap(image_id, attention_weights_dict, dataset, data_dir):
    # Find the file corresponding to the image_id
    file_name = image_id + '.gz'
    file_path = os.path.join(data_dir, file_name)
    
    # Extract tile positions and features
    array_row, array_col, _ = extract_tile_positions_and_features(file_path)
    
    # Get the attention weights for the image_id
    attention_weights = attention_weights_dict[image_id]
    
    # Ensure the attention weights are 1-dimensional and match the number of tiles
    if attention_weights.ndim != 1 or len(attention_weights) != len(array_row):
        raise ValueError("Mismatch in dimensions of attention weights and tile positions")
    
    # Create a DataFrame to store tile positions and corresponding attention weights
    attention_df = pd.DataFrame({
        'array_row': array_row,
        'array_col': array_col,
        'attention_weight': attention_weights
    })
    
    # Pivot the DataFrame to create a matrix for the heatmap
    attention_matrix = attention_df.pivot(index='array_row', columns='array_col', values='attention_weight')
    
    # Flip the attention matrix horizontally
    attention_matrix = np.fliplr(attention_matrix.values)
    
    # Plot the attention heatmap using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Attention Heatmap for Image ID: {image_id}')
    plt.xlabel('Tile Column')
    plt.ylabel('Tile Row')
    plt.gca().invert_yaxis()
    plt.show()

test_image_id = '2447'  # Replace with the actual image ID
data_dir = '/flashscratch/thiesa/img_features'
attention_weights_dict = attention_weights
# Plot the attention heatmap
plot_attention_heatmap(test_image_id, attention_weights_dict, dataset, data_dir)
