#jajajaja
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn import metrics
import scipy as sp

# Combine the datasets vertically
#combined_df = np.vstack((ctp_df_morpho_both_nos, stj_df_morpho_both_nos, yale_df_morpho_both_rms, stj_df_morpho_both_nrsts_nos))  

combined_df = filtered_array

# Define the number of features for each type
num_general_features = 15360  # Example: total number of general feature columns
num_eccentricity_features = 19  # Example: number of eccentricity columns
num_area_features = 19  # Example: number of area columns

# Calculate total number of features (excluding label column)
num_features = num_general_features + num_eccentricity_features + num_area_features

# Define percentile ranges for each feature set
pers_features = list(range(5, 100, 10))  # Example percentiles for general features
pers_eccentricity = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# Define the list of deciles to calculate for area
pers_area = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
PT = 0.1  # p-value threshold

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True)
aucs = []

# Perform 20 iterations of stratified k-fold cross-validation
for iteration in range(2):
    for train_index, test_index in skf.split(combined_df[:, :num_features], combined_df[:, num_features]):
        # Split the combined dataset into training and testing sets
        sampler_train = combined_df[train_index, :num_features]
        subtype_train = combined_df[train_index, num_features]
        
        sampler_test = combined_df[test_index, :num_features]
        subtype_test = combined_df[test_index, num_features]
        
        # Perform ANOVA (one-way) to test significance across three classes
        P = np.array([sp.stats.f_oneway(
            sampler_train[subtype_train == 0][:, i], 
            sampler_train[subtype_train == 1][:, i], 
            sampler_train[subtype_train == 2][:, i])[1] for i in range(num_features)])
        
        # Separate p-values for different feature types
        P_features = P[:num_general_features]  # P-values for general features
        P_eccentricity = P[num_general_features:num_general_features + num_eccentricity_features]  # P-values for eccentricity features
        P_area = P[num_general_features + num_eccentricity_features:]  # P-values for area features
        
        # Reshape the p-values for general features into matrix format (Pmat_features)
        Pmat_features = np.empty((len(pers_features), num_general_features // len(pers_features)))

        for idx, percentile in enumerate(pers_features):
            start_idx = idx * (num_general_features // len(pers_features))
            end_idx = (idx + 1) * (num_general_features // len(pers_features))
            Pmat_features[idx, :] = P_features[start_idx:end_idx]
        
        # Reshape the p-values for eccentricity into matrix format (Pmat_eccentricity)
        Pmat_eccentricity = np.empty((len(pers_eccentricity), len(pers_eccentricity)))

        for idx, percentile in enumerate(pers_eccentricity):
            Pmat_eccentricity[idx, :] = P_eccentricity[idx]  # Assuming one-to-one match
        
        # Reshape the p-values for area into matrix format (Pmat_area)
        Pmat_area = np.empty((len(pers_area), len(pers_area)))

        for idx, percentile in enumerate(pers_area):
            Pmat_area[idx, :] = P_area[idx]  # Assuming one-to-one match
        
        # Compute Wmat for each matrix
        Wmat_features = -1 * np.log(Pmat_features)
        Wmat_eccentricity = -1 * np.log(Pmat_eccentricity)
        Wmat_area = -1 * np.log(Pmat_area)

        # Apply feature selection based on p-value threshold for general features
        selected_features = np.where(P_features < PT)[0]  # Extract feature indices where the condition is true

        # Check if there are any selected features
        if selected_features.size > 0:
            # Train Logistic Regression model with L1 regularization for multiclass classification
            mlr = LR(penalty='l1', 
                     C=100, 
                     class_weight='balanced', 
                     solver='liblinear', 
                     max_iter=2000, 
                     multi_class='ovr').fit(sampler_train[:, selected_features], subtype_train)
            
            # Predict probabilities for each class
            probs = mlr.predict_proba(sampler_test[:, selected_features])
            
            # Compute multiclass AUC using one-vs-rest strategy
            auc = metrics.roc_auc_score(subtype_test, probs, multi_class='ovr', average='weighted')
            
            # Store the AUC value
            aucs.append(auc)
        else:
            print("No features selected based on the p-value threshold.")

# Output or analyze the results
print("Mean AUC over 20 iterations:", np.mean(aucs))
print("Standard Deviation of AUC:", np.std(aucs))
