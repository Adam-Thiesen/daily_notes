import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Assume array1, array2, array3, and array4 are your existing numpy arrays
# Extract column 3339 from each array and concatenate them into a single array
column_3339_all = np.concatenate([cog_feats[:, 3339], stj_nrsts_feats[:, 3339], stj_rms_feats[:, 3339], yale_rms_feats[:, 3339]])

# Calculate Z-scores for the concatenated data
z_scores = zscore(column_3339_all)
z_threshold = 3  # Common threshold for Z-score outliers
z_upper = np.mean(column_3339_all) + z_threshold * np.std(column_3339_all)
z_lower = np.mean(column_3339_all) - z_threshold * np.std(column_3339_all)

# Calculate IQR and thresholds
q1 = np.percentile(column_3339_all, 25)
q3 = np.percentile(column_3339_all, 75)
iqr = q3 - q1
iqr_threshold = 1.5  # Common multiplier for IQR outliers
iqr_upper = q3 + iqr_threshold * iqr
iqr_lower = q1 - iqr_threshold * iqr

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(column_3339_all, label='Combined Column 3339')

# Add lines for Z-score thresholds
plt.axhline(z_upper, color='r', linestyle='--', linewidth=1, label=f'Z-score Upper ({z_threshold}σ)')
plt.axhline(z_lower, color='r', linestyle='--', linewidth=1, label=f'Z-score Lower ({z_threshold}σ)')

# Add lines for IQR thresholds
plt.axhline(iqr_upper, color='b', linestyle='--', linewidth=1, label=f'IQR Upper (1.5 × IQR)')
plt.axhline(iqr_lower, color='b', linestyle='--', linewidth=1, label=f'IQR Lower (1.5 × IQR)')

# Labels and title
plt.xlabel('Combined Row Index')
plt.ylabel('Value')
plt.title('Values of Column 3339 Across All Arrays (Concatenated)')
plt.legend()
plt.show()
