import os
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import argparse
import seaborn as sns  # For histogram plotting with KDE

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

        # Ensure the slide_id strings match file names
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
                    features = (
                        df.iloc[:, self.start_col : self.start_col + self.num_features]
                          .apply(pd.to_numeric, errors='coerce')
                          .fillna(0)
                          .values
                    )
                    self.data.append(features)
                    self.tile_indices.append(df.index.to_list())
                    label = int(label_row['labels'].values[0])
                    self.labels.append(label)
                    self.image_ids.append(base_name)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
          - tile features for a single WSI: shape (num_tiles, num_features)
          - label: 0 or 1
          - image_id: identifier for the WSI
          - tile_indices: original tile indices
        """
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.image_ids[idx],
            self.tile_indices[idx],
        )

# -------------------------- MIL ATTENTION MODEL ----------------------------

class AttentionMIL(nn.Module):
    """
    An attention-based MIL model:
      1) Projects each tile into an embedding space.
      2) Learns attention scores for each tile.
      3) Uses the attention scores to create a weighted sum of tile embeddings.
      4) Passes that bag-level embedding into a final classification layer.
    """
    def __init__(self, input_dim=768, embed_dim=256, attn_hidden_dim=256):
        super().__init__()
        
        # Step 1: Embed each tile
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )
        
        # Step 2: Attention network to get attention logits
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1)  # produces a single score per tile
        )
        
        # Step 3: Classification head
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        x shape: [batch_size, num_tiles, input_dim]
        """
        # 1) Embed each tile
        embedded = self.embedding(x)  # [batch_size, num_tiles, embed_dim]
        
        # 2) Compute attention logits for each tile => [batch_size, num_tiles, 1]
        attn_logits = self.attn_net(embedded)

        # 3) Convert logits to normalized attention weights via softmax => [batch_size, num_tiles, 1]
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # 4) Weighted sum of tile embeddings => [batch_size, embed_dim]
        bag_repr = torch.sum(attn_weights * embedded, dim=1)
        
        # 5) Final classification => shape [batch_size, 1]
        logits = self.classifier(bag_repr).squeeze(dim=-1)
        
        # Return (logits, attn_weights) so we can optionally inspect them
        return logits, attn_weights.squeeze(-1)


# -------------------------- TRAINING & EVALUATION ---------------------------

def train(model, dataloader, criterion, optimizer, device, epochs=50, fold=0):
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels, _, _ in dataloader:
            inputs = inputs.to(device)    # shape [1, num_tiles, 768]
            labels = labels.to(device).float()
            labels = labels.view(-1)      # shape [1]

            optimizer.zero_grad()
            logits, _ = model(inputs)     # Only use the logits for loss computation
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        log_weight_stats(model, epoch+1, fold)

    return epoch_losses


def evaluate(model, dataloader, random_seed, output_file="mil3_metrics_output_kfold2.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_preds = []
    image_labels = []
    correct_images = []
    incorrect_images = []

    with torch.no_grad():
        for inputs, labels, image_ids, tile_indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.view(-1)
            outputs, _ = model(inputs)
            probs = torch.sigmoid(outputs)  # shape: [batch_size]

            # Collect predictions & labels
            image_preds.append(probs.cpu().numpy())
            image_labels.append(labels.cpu().numpy())

            # Single-sample logic for correct vs incorrect
            pred_label = (probs > 0.5).float().item()
            true_label = labels.item()
            if pred_label == true_label:
                correct_images.append(image_ids[0])
            else:
                incorrect_images.append(image_ids[0])

    image_preds = np.concatenate(image_preds)
    image_labels = np.concatenate(image_labels)

    # Metrics
    final_preds = (image_preds > 0.5).astype(int)
    accuracy = accuracy_score(image_labels, final_preds)
    auroc = roc_auc_score(image_labels, image_preds)
    
    report = classification_report(
        image_labels,
        final_preds,
        target_names=['EMBRYONAL', 'ALVEOLAR'],
        output_dict=True
    )
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']

    with open(output_file, "w") as f:
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"AUC: {auroc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Correct images: {correct_images}\n")
        f.write(f"Incorrect images: {incorrect_images}\n")
        f.write("Learning rate: 0.0001\n")       # <-- Match optimizer
        f.write(f"Random state: {random_seed}\n")  # <-- Use actual seed

    return accuracy, auroc, precision, recall

def log_weight_stats(model, epoch, fold):
    """
    Logs statistics of the FIRST layer in the embedding module 
    (nn.Linear(768, 256)).
    """
    # Access the first linear layer in model.embedding
    first_layer = model.embedding[0]
    W = first_layer.weight.detach().cpu().numpy()
    w_mean = np.mean(W)
    w_std  = np.std(W)
    w_min  = np.min(W)
    w_max  = np.max(W)
    log_str = (f"Epoch {epoch}: Mean={w_mean:.6f}, Std={w_std:.6f}, "
               f"Min={w_min:.6f}, Max={w_max:.6f}\n")
    
    # Write to per-fold file
    file_name = f"mil2_weight_stats_fold_{fold+1}.txt"
    with open(file_name, "a") as f:
        f.write(log_str)
    
    print(log_str)


def main(random_seed):
    # Reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Read labels
    labels_df = pd.read_csv('/flashscratch/thiesa/Pytorch5/labels.csv')
    label_mapping = {'EMBRYONAL': 0, 'ALVEOLAR': 1}
    labels_df['labels'] = labels_df['labels'].map(label_mapping)
    print(labels_df.head())

    # 2) Load dataset
    data_dir = '/flashscratch/thiesa/ctransapth_20x_features'
    dataset = RhabdomyosarcomaDataset(data_dir, labels_df)

    # 3) Compute class weights
    class_counts = np.bincount(dataset.labels)
    majority_class_count = max(class_counts)
    minority_class_count = min(class_counts)
    pos_weight_value = majority_class_count / minority_class_count
    print(f"pos_weight_value = {pos_weight_value:.2f}")
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    # 4) Define loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5) Identify unique WSIs for stratification
    image_ids = np.array(dataset.image_ids)
    unique_image_ids = np.unique(image_ids)
    unique_labels = [
        dataset.labels[np.where(image_ids == img_id)[0][0]] for img_id in unique_image_ids
    ]

    # 6) Group K-fold by patient_id
    patient_ids = labels_df.set_index('slide_id').loc[unique_image_ids, 'patient_id'].values
    stratified_group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)

    fold_results = []
    all_fold_losses = []

    # 7) Cross-validation
    for fold, (train_index, test_index) in enumerate(
        stratified_group_kfold.split(unique_image_ids, unique_labels, groups=patient_ids)
    ):
        print(f"========== Fold {fold+1} ==========")
        train_image_ids = unique_image_ids[train_index]
        test_image_ids = unique_image_ids[test_index]

        train_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in train_image_ids]
        test_indices  = [i for i, img_id in enumerate(dataset.image_ids) if img_id in test_image_ids]

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # ---------- Use the Attention-based MIL Model -----------
        model = AttentionMIL(input_dim=768, embed_dim=256, attn_hidden_dim=256)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # Train
        fold_losses = train(
            model, train_loader, criterion, optimizer, device,
            epochs=50, fold=fold
        )
        all_fold_losses.append(fold_losses)

        # ---------------- PRINT EXTENDED WEIGHT & BIAS STATISTICS ---------------
        embedding_layer = model.embedding[0]  # shape: (256, 768)
        w = embedding_layer.weight.detach().cpu().numpy()
        b = embedding_layer.bias.detach().cpu().numpy()

        # Flatten the weights for distribution stats
        w_flat = w.flatten()

        # Overall weight distribution stats
        w_mean = np.mean(w_flat)
        w_std = np.std(w_flat)
        w_min = np.min(w_flat)
        w_max = np.max(w_flat)
        num_zero = np.sum(w_flat == 0)
        near_zero_thresh = 1e-5
        num_near_zero = np.sum(np.abs(w_flat) < near_zero_thresh)
        large_thresh = 1.0
        num_large = np.sum(np.abs(w_flat) > large_thresh)

        # Bias stats
        b_mean = np.mean(b)
        b_std = np.std(b)
        b_min = np.min(b)
        b_max = np.max(b)

        # L1-based feature importance
        feature_importance = np.sum(np.abs(w), axis=0)  # shape [768]

        plt.figure(figsize=(8, 6))
        sns.histplot(feature_importance, bins=50, kde=True)
        plt.title(f"Distribution of L1 Norms per Feature (Fold {fold+1})")
        plt.xlabel("L1 Norm of Feature Weights")
        plt.ylabel("Density")
        plt.savefig(f"mil3_feature_importance_distribution_fold_{fold+1}.png")
        # plt.show()
        ranking = np.argsort(feature_importance)[::-1]  # descending order

        # Write stats to text file
        stats_filename = f"weights_bias_stats_mil10_fold{fold+1}.txt"
        with open(stats_filename, "w") as stats_file:
            stats_file.write(f"FOLD {fold+1} WEIGHT & BIAS STATISTICS\n")
            stats_file.write("---------------------------------------\n\n")
            # Overall weight distribution
            stats_file.write("WEIGHTS:\n")
            stats_file.write(f"  Shape:         {w.shape}\n")
            stats_file.write(f"  Mean:          {w_mean:.6f}\n")
            stats_file.write(f"  Std:           {w_std:.6f}\n")
            stats_file.write(f"  Min:           {w_min:.6f}\n")
            stats_file.write(f"  Max:           {w_max:.6f}\n")
            stats_file.write(f"  Num Exactly 0: {num_zero}\n")
            stats_file.write(f"  Num |w|<{near_zero_thresh}: {num_near_zero}\n")
            stats_file.write(f"  Num |w|>{large_thresh}: {num_large}\n\n")

            # Bias distribution
            stats_file.write("BIASES:\n")
            stats_file.write(f"  Shape:    {b.shape}\n")
            stats_file.write(f"  Mean:     {b_mean:.6f}\n")
            stats_file.write(f"  Std:      {b_std:.6f}\n")
            stats_file.write(f"  Min:      {b_min:.6f}\n")
            stats_file.write(f"  Max:      {b_max:.6f}\n\n")

            # Print top 5 & bottom 5 features by L1 norm
            stats_file.write("FEATURE IMPORTANCE (L1 norm, descending):\n")
            for i in range(5):
                feat_idx = ranking[i]
                stats_file.write(
                    f"  Top {i+1}: Feature {feat_idx} -> {feature_importance[feat_idx]:.4f}\n"
                )
            stats_file.write("...\n")
            for i in range(5):
                feat_idx = ranking[-(i+1)]
                stats_file.write(
                    f"  Bottom {i+1}: Feature {feat_idx} -> {feature_importance[feat_idx]:.4f}\n"
                )

        # Plot distribution of embedding-layer weights
        plt.figure(figsize=(8, 6))
        sns.histplot(w_flat, bins=50, kde=True)
        plt.title(f"Distribution of Embedding Layer Weights (Fold {fold+1})")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.savefig(f"mil10_embedding_weights_distribution_fold_{fold+1}.png")
        # plt.show()

        print(f"\n[Fold {fold+1}] First Layer Weight Shape: {w.shape}")
        print(f"[Fold {fold+1}] First Layer Bias Shape:   {b.shape}")
        print(f"[Fold {fold+1}] First layer bias (first 5): {b[:5]}")
        print(f"[Fold {fold+1}] See '{stats_filename}' for full weight/bias stats.\n")
        # --------------------------------------------------------------------------

        # Evaluate
        accuracy, auroc, precision, recall = evaluate(
            model, test_loader, random_seed,
            output_file=f"mil10_metrics_output_fold{fold+1}.txt"
        )
        print(f"Fold {fold+1} --> Accuracy={accuracy:.4f}, AUC={auroc:.4f}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}")
        
        fold_results.append((accuracy, auroc, precision, recall))

    # 8) Plot training loss
    plt.figure(figsize=(10, 6))
    for fold_idx, losses in enumerate(all_fold_losses):
        plt.plot(losses, label=f"Fold {fold_idx + 1}")
    plt.title("Training Loss per Epoch (Attention MIL)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss_mil10.png")
    # plt.show()

    # 9) Average across folds
    avg_accuracy = np.mean([res[0] for res in fold_results])
    avg_auroc = np.mean([res[1] for res in fold_results])
    avg_precision = np.mean([res[2] for res in fold_results])
    avg_recall = np.mean([res[3] for res in fold_results])

    print(f"\n\nCross-Validation Results:")
    print(f"Avg Accuracy:  {avg_accuracy:.4f}")
    print(f"Avg AUC:       {avg_auroc:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")
    print(f"Avg Recall:    {avg_recall:.4f}")

    with open("mil10_cross_validation_metrics.txt", "w") as f:
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"Average AUC: {avg_auroc:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write("Learning rate = 1e-4\n")
        f.write("Model = Attention-based MIL\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention-based MIL approach with 5-fold cross-validation.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for cross-validation")
    args = parser.parse_args()
    
    main(args.seed)
