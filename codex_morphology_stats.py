import pandas as pd
# This is to normalize the data per region/tif
dfz = pd.DataFrame()

for region in df_filt.unique_region.unique():
    df_reg = df_filt[df_filt.unique_region == region]
    df_reg_norm = sp.pp.format(
        data=df_reg, 
        list_out= ['convex_area', 'axis_major_length', 'axis_minor_length',  "label"], # list of features to remove
        list_keep = ["DAPI",'x','y', 'area','region_num',"unique_region", "eccentricity", "perimeter"], # list of meta information that you would like to keep but don't want to normalize
        method = "zscore") # choose from "zscore", "double_zscore", "MinMax", "ArcSin"
    dfz = pd.concat([dfz,df_reg_norm], axis = 0)

dfz.shape

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame
# Columns: 'p21', 'CD31', 'aSMA', 'eccentricity'

df = dfz

# Calculate 70th percentile thresholds for each marker
p21_threshold = df['p21'].quantile(0.8)
CD31_threshold = df['CD31'].quantile(0.5)
aSMA_threshold = df['aSMA'].quantile(0.5)

# Print thresholds for verification
print(f"p21 threshold (70th percentile): {p21_threshold}")
print(f"CD31 threshold (70th percentile): {CD31_threshold}")
print(f"aSMA threshold (70th percentile): {aSMA_threshold}")

# Filter cells
group1 = df[(df['p21'] > p21_threshold) & 
            (df['CD31'] <= CD31_threshold) & 
            (df['aSMA'] <= aSMA_threshold)]

group2 = df[~((df['p21'] > p21_threshold) & 
              (df['CD31'] <= CD31_threshold) & 
              (df['aSMA'] <= aSMA_threshold))]

# Get eccentricity values
eccentricity_group1 = group1['eccentricity']
eccentricity_group2 = group2['eccentricity']

# Perform statistical test
stat, p_value = stats.mannwhitneyu(eccentricity_group1, eccentricity_group2, alternative='two-sided')

print(f"Mann-Whitney U Test: Statistic={stat}, p-value={p_value}")

import numpy as np

# Convert to numpy arrays
eccentricity_group1 = np.array(eccentricity_group1)
eccentricity_group2 = np.array(eccentricity_group2)

# Re-plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=[eccentricity_group1, eccentricity_group2], notch=True)
plt.xticks([0, 1], ['High p21, Not High CD31/aSMA', 'All Other Cells'])
plt.ylabel('Eccentricity')
plt.title('Comparison of Eccentricity Values')
plt.show()
