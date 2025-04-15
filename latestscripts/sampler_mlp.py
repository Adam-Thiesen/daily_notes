import os
import gzip
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report,
    roc_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# 1) DATASET
##############################################################################
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
                    features = df.iloc[:, self.start_col:self.start_col + self.num_features] \
                                   .apply(pd.to_numeric, errors='coerce') \
                                   .fillna(0).values
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
            features: Tensor of shape [num_tiles, 768]
            label: Long tensor with label
            image_id: Unique slide/image identifier
            tile_indices: Original tile indices
        """
        features = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        image_id = self.image_ids[idx]
        tile_idx = self.tile_indices[idx]
        return features, label, image_id, tile_idx


##############################################################################
# 2) PERCENTILE AGGREGATOR
##############################################################################
class PercentileAggregator(nn.Module):
    """
    Non-differentiable aggregator that sorts tile-wise features
    and extracts approximate percentiles across tiles for each dimension.
    """
    def __init__(self, embed_dim=256, n_percentiles=10, min_pct=5, max_pct=95):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_percentiles = n_percentiles
        self.min_pct = min_pct
        self.max_pct = max_pct

        # Build a list of fractions, e.g. [0.05, 0.15, 0.25, ... up to 0.95]
        self.percentile_fracs = np.linspace(min_pct / 100.0, max_pct / 100.0, n_percentiles)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, num_tiles, embed_dim]
        Returns:
            aggregator_output: shape [batch_size, embed_dim * n_percentiles]
        """
        b, n, d = x.shape
        outputs = []

        # Transpose to [batch_size, embed_dim, num_tiles] for easier dimension-wise loop
        x_t = x.transpose(1, 2).contiguous()  # [b, d, n]

        for dim_i in range(d):
            # shape [b, n]
            dim_data = x_t[:, dim_i, :]

            # Sort along the tile dimension => not differentiable
            sorted_dim, _ = torch.sort(dim_data, dim=1)

            percentile_values = []
            for frac in self.percentile_fracs:
                idx_float = frac * (n - 1)    
                idx_lower = int(np.floor(idx_float))
                idx_upper = int(np.ceil(idx_float))

                val_lower = sorted_dim[:, idx_lower]
                val_upper = sorted_dim[:, idx_upper]

                weight_upper = idx_float - idx_lower
                val = val_lower * (1.0 - weight_upper) + val_upper * weight_upper
                percentile_values.append(val.unsqueeze(1))  # shape [b, 1]

            # shape [b, n_percentiles]
            dim_percentiles = torch.cat(percentile_values, dim=1)
            outputs.append(dim_percentiles)

        # Concatenate along dimension=1 => [b, d * n_percentiles]
        aggregator_output = torch.cat(outputs, dim=1)
        return aggregator_output


##############################################################################
# 3) MODEL
##############################################################################
class PercentileModelNoProjection(nn.Module):
    """
    1) Applies a percentile aggregator directly on the original tile features (e.g., 768-dim).
    2) Feeds aggregated percentile features into an MLP for classification.
    """
    def __init__(
        self,
        input_dim=768,
        n_percentiles=10,
        min_pct=5,
        max_pct=95,
        mlp_hidden=256,
        dropout=0
    ):
        super().__init__()

        # Use the original 768-dim features in the aggregator:
        self.aggregator = PercentileAggregator(
            embed_dim=input_dim,
            n_percentiles=n_percentiles,
            min_pct=min_pct,
            max_pct=max_pct
        )

        # After aggregator => shape is [batch_size, input_dim * n_percentiles]
        self.fc = nn.Sequential(
            nn.Linear(input_dim * n_percentiles, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)  # output a single logit
        )

    def forward(self, x):
        """
        x: [batch_size, num_tiles, input_dim]
        """
        # 1) Aggregate percentiles across tiles => shape [b, input_dim * n_percentiles]
        x = self.aggregator(x)

        # 2) MLP => shape [b, 1]
        x = self.fc(x)
        return x.squeeze(1)  # => [batch_size]


##############################################################################
# 4) TRAINING LOOP
##############################################################################
def train(model, dataloader, criterion, optimizer, device, epochs=20, fold=0):
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels, _, _ in dataloader:
            inputs = inputs.to(device)    # shape [1, num_tiles, 768]
            labels = labels.to(device).float()
            labels = labels.view(-1)      # shape [1]

            optimizer.zero_grad()
            outputs = model(inputs)       # shape [1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        log_weight_stats(model, epoch+1, fold)

    return epoch_losses


def log_weight_stats(model, epoch, fold):
    first_layer = model.fc[0]
    W = first_layer.weight.detach().cpu().numpy()
    w_mean = np.mean(W)
    w_std  = np.std(W)
    w_min  = np.min(W)
    w_max  = np.max(W)
    log_str = f"Epoch {epoch}: Mean={w_mean:.6f}, Std={w_std:.6f}, Min={w_min:.6f}, Max={w_max:.6f}\n"
    
    # Specify a file name that is unique per fold
    file_name = f"weight_stats_fold_{fold+1}.txt"
    
    # Open in append mode so that each epoch's log is added to the file
    with open(file_name, "a") as f:
        f.write(log_str)
    
    # Optionally, still print the stats to the console
    print(log_str)

##############################################################################
# 5) EVALUATION
##############################################################################
def evaluate(model, dataloader, output_file=None):
    model.eval()
    device = next(model.parameters()).device

    image_preds = []
    image_labels = []
    correct_images = []
    incorrect_images = []

    with torch.no_grad():
        for inputs, labels, image_ids, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            labels = labels.view(-1)

            outputs = model(inputs)         # shape [1]
            probs = torch.sigmoid(outputs)  # shape [1]

            pred_label = (probs > 0.5).float().item()
            true_label = labels.item()

            image_preds.append(probs.cpu().numpy())
            image_labels.append(labels.cpu().numpy())

            # Track correct/incorrect
            if pred_label == true_label:
                correct_images.append(image_ids[0])
            else:
                incorrect_images.append(image_ids[0])

    # Convert to arrays
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
    recall    = report['weighted avg']['recall']

    # Optional: write to file
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(f"Accuracy: {accuracy*100:.2f}%\n")
            f.write(f"AUC: {auroc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Correct images: {correct_images}\n")
            f.write(f"Incorrect images: {incorrect_images}\n")

    return accuracy, auroc, precision, recall, image_preds, final_preds, image_labels


##############################################################################
# 6) MAIN: 5-Fold Cross-Validation Example
##############################################################################
def main(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- (A) Load labels
    labels_df = pd.read_csv("/flashscratch/thiesa/Pytorch5/labels.csv")
    label_mapping = {"EMBRYONAL": 0, "ALVEOLAR": 1}
    labels_df["labels"] = labels_df["labels"].map(label_mapping)
    print(labels_df.head())

    # -- (B) Initialize dataset
    data_dir = "/flashscratch/thiesa/ctransapth_20x_features"
    dataset = RhabdomyosarcomaDataset(data_dir, labels_df)

    # -- (C) Class counts => pos_weight for BCE
    class_counts = np.bincount(dataset.labels)
    majority_class_count = max(class_counts)
    minority_class_count = min(class_counts)
    pos_weight_value = majority_class_count / float(minority_class_count)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -- (D) Unique IDs, labels, patient_ids for StratifiedGroupKFold
    image_ids = np.array(dataset.image_ids)
    unique_image_ids = np.unique(image_ids)
    unique_labels = [
        dataset.labels[np.where(image_ids == img_id)[0][0]]
        for img_id in unique_image_ids
    ]
    patient_ids = labels_df.set_index("slide_id").loc[unique_image_ids, "patient_id"].values

    skf = StratifiedGroupKFold(n_splits=5, random_state=random_seed, shuffle=True)

    fold_results = []
    all_fold_losses = []
    fold_fprs = []
    fold_tprs = []
    accum_cm = np.zeros((2, 2), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(unique_image_ids, unique_labels, groups=patient_ids)
    ):
        print(f"\n===== Training fold {fold + 1} =====")

        train_image_ids = unique_image_ids[train_idx]
        test_image_ids  = unique_image_ids[test_idx]

        # Map back to indices in the full dataset
        train_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in train_image_ids]
        test_indices  = [i for i, img_id in enumerate(dataset.image_ids) if img_id in test_image_ids]

        train_dataset = Subset(dataset, train_indices)
        test_dataset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # -- (E) Create the model (no projection)
        model = PercentileModelNoProjection(
            input_dim=768,
            n_percentiles=10,
            min_pct=5,
            max_pct=95,
            mlp_hidden=256,
            dropout=0
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # -- (F) Train
        fold_losses = train(model, train_loader, criterion, optimizer, device, epochs=50, fold=fold)
        all_fold_losses.append(fold_losses)

        # -------------------------- EXTENDED WEIGHT & BIAS STATISTICS ---------------------------
        # The first layer of the MLP is model.fc[0] => nn.Linear(768 * n_percentiles, mlp_hidden)
        first_layer = model.fc[0]
        W = first_layer.weight.detach().cpu().numpy()  # shape [mlp_hidden, (768 * n_percentiles)]
        b = first_layer.bias.detach().cpu().numpy()    # shape [mlp_hidden]

        # Flatten the weights to compute distribution stats
        W_flat = W.flatten()
        w_mean = np.mean(W_flat)
        w_std  = np.std(W_flat)
        w_min  = np.min(W_flat)
        w_max  = np.max(W_flat)

        num_zero = np.sum(W_flat == 0)
        near_zero_thresh = 1e-5
        num_near_zero = np.sum(np.abs(W_flat) < near_zero_thresh)
        large_thresh = 1.0
        num_large = np.sum(np.abs(W_flat) > large_thresh)

        # Bias stats
        b_mean = np.mean(b)
        b_std  = np.std(b)
        b_min  = np.min(b)
        b_max  = np.max(b)

        # Calculate L1-based feature importance across hidden units
        feature_importance = np.sum(np.abs(W), axis=0)  # shape [(768 * n_percentiles)]
        ranking = np.argsort(feature_importance)[::-1]  # descending order

        # Calculate L1-based feature importance across hidden units
        feature_importance = np.sum(np.abs(W), axis=0)  # shape [(768 * n_percentiles)]

        # Plot the distribution of the L1 norms (feature importance values)
        plt.figure(figsize=(8, 6))
        sns.histplot(feature_importance, bins=50, kde=True)
        plt.title(f"Distribution of L1 Norms (Feature Importance) (Fold {fold+1})")
        plt.xlabel("L1 Norm")
        plt.ylabel("Frequency")
        plt.savefig(f"l1_5_norm_distribution_fold_{fold+1}.png")
        #plt.show()


        # Write extended stats to a separate .txt file for each fold
        stats_filename = f"mlp5_weights_bias_stats_fold{fold+1}.txt"
        with open(stats_filename, "w") as sf:
            sf.write(f"FOLD {fold+1} WEIGHT & BIAS STATISTICS\n")
            sf.write("---------------------------------------\n\n")
            sf.write("WEIGHTS:\n")
            sf.write(f"  Shape           : {W.shape}\n")
            sf.write(f"  Mean            : {w_mean:.6f}\n")
            sf.write(f"  Std             : {w_std:.6f}\n")
            sf.write(f"  Min             : {w_min:.6f}\n")
            sf.write(f"  Max             : {w_max:.6f}\n")
            sf.write(f"  Num Exactly 0   : {num_zero}\n")
            sf.write(f"  Num |w|<{near_zero_thresh} : {num_near_zero}\n")
            sf.write(f"  Num |w|>{large_thresh}     : {num_large}\n\n")

            sf.write("BIASES:\n")
            sf.write(f"  Shape    : {b.shape}\n")
            sf.write(f"  Mean     : {b_mean:.6f}\n")
            sf.write(f"  Std      : {b_std:.6f}\n")
            sf.write(f"  Min      : {b_min:.6f}\n")
            sf.write(f"  Max      : {b_max:.6f}\n\n")

            # Top 5 and bottom 5 by L1 norm
            sf.write("FEATURE IMPORTANCE (L1 norm, descending):\n")
            for i in range(5):
                idx = ranking[i]
                sf.write(f"  Top {i+1} idx {idx}: {feature_importance[idx]:.4f}\n")
            sf.write("...\n")
            for i in range(5):
                idx = ranking[-(i+1)]
                sf.write(f"  Bottom {i+1} idx {idx}: {feature_importance[idx]:.4f}\n")

        # Plot the distribution of the flattened weights
        plt.figure(figsize=(8, 6))
        sns.histplot(W_flat, bins=50, kde=True)
        plt.title(f"Distribution of First Layer Weights (Fold {fold+1})")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.savefig(f"weights5_distribution_fold_{fold+1}.png")
        #plt.show()

        # Minimal console print
        print(f"\n[Fold {fold+1}] First Layer Weight Shape: {W.shape}")
        print(f"[Fold {fold+1}] First Layer Bias Shape:   {b.shape}")
        print(f"[Fold {fold+1}] First layer bias (first 5): {b[:5]}")
        print(f"[Fold {fold+1}] See '{stats_filename}' for full weight/bias stats.\n")
        # ----------------------------------------------------------------------------------------

        # -- (G) Evaluate
        acc, roc_auc, prec, rec, probs, bin_preds, true_labels = evaluate(
            model, test_loader, output_file=f"mlp3_deep_sampler_percentile_fold{fold+1}.txt"
        )
        fold_results.append((acc, roc_auc, prec, rec))
        print(f"Fold {fold+1} Accuracy: {acc:.4f} | AUC: {roc_auc:.4f} "
              f"| Precision: {prec:.4f} | Recall: {rec:.4f}")

        # Confusion matrix
        cm = confusion_matrix(true_labels, bin_preds)
        accum_cm += cm

        # FPR, TPR for ROC curve
        fpr, tpr, _ = roc_curve(true_labels, probs)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)

    # -- (H) Plot training loss
    plt.figure(figsize=(10,6))
    for i, losses in enumerate(all_fold_losses):
        plt.plot(losses, label=f"Fold {i+1}")
    plt.title("Training Loss per Epoch (Percentile Model)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("mlp5_percentile_model_training_loss.png")
    #plt.show()

    # -- (I) Compute average CV metrics
    avg_acc   = np.mean([r[0] for r in fold_results])
    avg_auc   = np.mean([r[1] for r in fold_results])
    avg_prec  = np.mean([r[2] for r in fold_results])
    avg_rec   = np.mean([r[3] for r in fold_results])

    print(f"\n===== 5-Fold Results =====")
    print(f"Average Accuracy : {avg_acc:.4f}")
    print(f"Average AUC      : {avg_auc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Average Recall   : {avg_rec:.4f}")

    # -- (J) ROC Curves
    plt.figure(figsize=(7,6))
    for i, (fpr, tpr) in enumerate(zip(fold_fprs, fold_tprs)):
        auc_i = fold_results[i][1]
        plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC={auc_i:.2f})")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Each Fold (Percentile Model)")
    plt.legend(loc="lower right")
    plt.savefig("percentile5_model_roc_curves.png")
    #plt.show()

    # -- (K) Average confusion matrix
    avg_cm = accum_cm / 5.0
    plt.figure(figsize=(5,4))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["ALVEOLAR (Pred)", "EMBRYONAL (Pred)"],
                yticklabels=["ALVEOLAR (True)", "EMBRYONAL (True)"])
    plt.title("Average Confusion Matrix (Percentile Model, 5-Fold)")
    #plt.savefig("percentile_model_confusion_matrix.png")
    #plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
