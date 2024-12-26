#JAJAJAJAJA

#read in segmentation csv files
#Read and concatenate the csv files (outputs from the cell segmentation algorithms). 
df_seg = sp.pp.read_segdf(
    segfile_list = [ # list of segmented files
    output_dir + "spot1_mesmer_result.csv",
    output_dir + "spot2_mesmer_result.csv",
    output_dir + "spot3_mesmer_result.csv",
    output_dir + "spot4_mesmer_result.csv",
    output_dir + "spot5_mesmer_result.csv",
    output_dir + "spot6_mesmer_result.csv",
    output_dir + "spot8_mesmer_result.csv",
    output_dir + "spot9_mesmer_result.csv",
    output_dir + "spot11_mesmer_result.csv",
    output_dir + "spot12_mesmer_result.csv",
    output_dir + "spot13_mesmer_result.csv",
    output_dir + "spot14_mesmer_result.csv",
    output_dir + "spot15_mesmer_result.csv",
    output_dir + "spot16_mesmer_result.csv",
    output_dir + "spot17_mesmer_result.csv",
    output_dir + "spot18_mesmer_result.csv",
    output_dir + "spot19_mesmer_result.csv",
    output_dir + "spot20_mesmer_result.csv",
    output_dir + "spot21_mesmer_result.csv",
    output_dir + "spot22_mesmer_result.csv",
    output_dir + "spot23_mesmer_result.csv",
    output_dir + "spot24_mesmer_result.csv",
    output_dir + "spot25_mesmer_result.csv",
    output_dir + "spot26_mesmer_result.csv",
    output_dir + "spot27_mesmer_result.csv",
    output_dir + "spot28_mesmer_result.csv",
    output_dir + "spot29_mesmer_result.csv",
    output_dir + "spot30_mesmer_result.csv"
],
    seg_method = 'mesmer',
    region_list = [
    "reg0001", "reg0002", "reg0003", "reg0004", "reg0005", 
    "reg0006", "reg0008", "reg0009", "reg0011", "reg0012", "reg0013", "reg0014", "reg0015", 
    "reg0016", "reg0017", "reg0018", "reg0019", "reg0020", 
    "reg0021", "reg0022", "reg0023", "reg0024", "reg0025", 
    "reg0026", "reg0027", "reg0028", "reg0029", "reg0030"]
,
)

#Get the shape of the data
print(df_seg.shape)

#See what it looks like
df_seg.head()

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
