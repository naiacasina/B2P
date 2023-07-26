# This file follows the sixth approach using as model inputs Matthew's model's outputs as well
# And builds different versions of the best models in Rwanda: dropping schools, health centers, GDP...

import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns
import os
import geopandas as gpd

folder = "Rwanda"
country = "rwanda"
approach = 'sixth'
drop = "elevation"  # also: "primary", "secondary", "health_centers", "semi_urban", "bridges", "population", "gdp"

os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")

# Import train-test dataframe
data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')
data = data.dropna()
inf_rows = data.isin([np.inf, -np.inf]).any(axis=1)
# Filter out rows with inf values
data = data[~inf_rows].copy()

data['elev_perc_dif'] = data['elev_p75'] - data['elev_p25']

# Assuming your DataFrame is called 'data'
if drop=="none":
    X = data.drop(['label', 'geometry'], axis=1)
elif drop=="primary":
    X = data.drop(['label', 'geometry', 'delta_time_df_primary_schools', 'max_time_df_primary_schools'], axis=1)
elif drop=="secondary":
    X = data.drop(['label', 'geometry', 'delta_time_df_secondary_schools', 'max_time_df_secondary_schools'], axis=1)
elif drop=="schools":
    X = data.drop(['label', 'geometry',
    'delta_time_df_primary_schools',
    'max_time_df_primary_schools',
    'delta_time_df_secondary_schools',
    'max_time_df_secondary_schools'], axis=1)
elif drop=="health_centers":
    X = data.drop(['label', 'geometry', 'delta_time_df_health_centers', 'max_time_df_health_centers'], axis=1)
elif drop=="semi_urban":
    X = data.drop(['label', 'geometry',
    'delta_time_df_semi_dense_urban',
    'max_time_df_semi_dense_urban'], axis=1)
elif drop=="bridges":
    X = data.drop(['label', 'geometry', 'nearest_distance_bridge'], axis=1)
elif drop=="population":
    X = data.drop(['label', 'geometry','pop_total'], axis=1)
elif drop=="gdp":
    X = data.drop(['label', 'geometry','max_gdp', 'mean_gdp'], axis=1)
elif drop=="elevation":
    X = data.drop(['label', 'geometry', 'elevation_difference', 'elev_p25', 'elev_p50', 'elev_p75',
    'terrain_ruggedness'], axis=1)

y = data['label']  # Target variable

# Normalize the numerical features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

test_prop = 0.2
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_prop, random_state=42)
# Define the custom scoring metric
scoring = {'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score)}
scoring = make_scorer(f1_score)


# Define classifiers with their respective hyperparameter spaces for grid search
classifiers = [
    ('XGBoost', xgb.XGBClassifier(),
     {'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01],
      'subsample': [0.8, 1.0], 'gamma': [0, 0.1, 0.2]}),
    ('Random Forest', RandomForestClassifier(),
     {'n_estimators': [100, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5],
      'min_samples_leaf': [1, 3], 'max_features': ['sqrt', 'log2']})
]


performance = {'Classifier': [], 'Accuracy': [], 'Recall': [], 'Precision': []}
best_params = {}

# Iterate over the classifiers, perform cross-validation, and evaluate each model
for name, clf, param_grid in classifiers:
    if name == 'XGBoost':
        # Calculate class weights for XGBoost
        class_weights = len(y_train) / (2 * np.bincount(y_train))
        weight_ratio = class_weights[1] / class_weights[0]
        clf.set_params(scale_pos_weight=weight_ratio)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring=scoring, refit='f1_score')
    grid_search.fit(X_train, y_train)

    # Use the best estimator from the grid search
    best_clf = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_

    # Perform cross-validation on the best estimator
    cv_scores = cross_val_score(best_clf, X_train, y_train, cv=5)

    # Make predictions on the test set
    y_pred = best_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    performance['Classifier'].append(name)
    performance['Accuracy'].append(accuracy)
    performance['Recall'].append(recall)
    performance['Precision'].append(precision)

    print(f"Classifier: {name}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Recall: {recall}")
    print(f"Test Precision: {precision}")
    print("-------------------------------------")

# Convert performance dictionary into a df
performance_df = pd.DataFrame(performance)
# Save the performance_df dictionary to a pickle file
with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped_{drop}.pkl', 'wb') as f:
    pickle.dump(performance_df, f)

with open(f'Saved data/best_params_{approach}_test_size_{test_prop}_dropped_{drop}.pkl', 'wb') as f:
    pickle.dump(best_params, f)

# ----- Plotting feature importances -----
with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped_{drop}.pkl', 'rb') as f:
    performance_df = pickle.load(f)
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}_dropped_{drop}.pkl', 'rb') as f:
    best_params = pickle.load(f)
# Set a color palette
colors = sns.color_palette('pastel')

# Create a figure and subplots
fig, axs = plt.subplots(len(classifiers), 1, figsize=(8, (len(classifiers)*4 )))

# Iterate over the classifiers
for i, (name, clf, param_grid) in enumerate(classifiers):
    # Exclude the SVM, KNN, and Neural Network classifiers
    # Get the feature importances for the current classifier
    try:
        # Get the best parameters for the current classifier
        best_params_clf = best_params[name]

        # Create an instance of the classifier with the best parameters
        clf_best = clf.set_params(**best_params_clf)
        clf_best.fit(X_train, y_train)
        feature_importances = clf_best.feature_importances_
        feature_names = X.columns

        # Plot feature importances in the corresponding subplot
        axs[i].bar(feature_names, feature_importances, color=colors)
        axs[i].set_xlabel('Features' if i == len(classifiers)-1 else '')
        axs[i].set_ylabel('Importance')
        axs[i].set_title(f'Feature Importances - {name} - dropped {drop}')
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].grid(axis='y', linestyle='--', alpha=0.5)

        # Omit x labels in the top subplot
        if i != len(classifiers)-1 :
            axs[i].set_xticklabels([])
    except AttributeError:
        # Skip classifiers without feature importances
        continue

# Adjust spacing between subplots
plt.tight_layout()
# Save the figure
plt.savefig(f'ML/Performance plots/feature_importances_{approach}_test_size_{test_prop}_dropped_{drop}.png')
# Show the plot
plt.show()


# ---------- Performance plots -----------

# ---- Performance plot for XGBoost and Random Forest ----
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(4, 4))

# Plot accuracy, recall, and precision
ax.plot(performance_df['Classifier'], performance_df['Accuracy'], marker='o', label='Accuracy')
ax.plot(performance_df['Classifier'], performance_df['Recall'], marker='o', label='Recall')
ax.plot(performance_df['Classifier'], performance_df['Precision'], marker='o', label='Precision')

# Set plot labels and title
ax.set_xlabel('Methods')
ax.set_ylabel('Scores')
ax.set_title(f'Performance -- Dropped {drop}')

# Customize plot appearance
ax.legend()
ax.grid(True)

# Set the y-limits
ax.set_ylim(0.3, 0.9)  # Adjust the values as needed

# Save the figure with higher resolution (e.g., DPI = 300)
dpi = 300  # Adjust the DPI value as needed
plt.savefig(f'ML/Performance plots/performance_plot_dropped_{drop}.png', dpi=dpi)

# Show the plot
plt.show()



# ----------- Performance plots all together ---------
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# List of drop variables
drop_variables = ['primary', 'secondary', 'schools', 'health_centers', 'semi_urban', 'bridges', 'population', 'gdp', 'elevation']

# Calculate the number of rows and columns needed for subplots
num_plots = len(drop_variables)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols

# Create a figure with subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 4*num_rows), sharex='all', sharey='all')

# Loop through each drop variable and plot the corresponding performance metrics
for idx, drop_var in enumerate(drop_variables):
    # Load the performance data for the specific drop variable
    with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped_{drop_var}.pkl', 'rb') as f:
        performance_df = pickle.load(f)

    # Calculate the row and column indices for the current subplot
    row_idx = idx // num_cols
    col_idx = idx % num_cols

    # Plot recall and precision for the current drop variable
    axs[row_idx, col_idx].plot(performance_df['Classifier'], performance_df['Recall'], marker='o', label='Recall')
    axs[row_idx, col_idx].plot(performance_df['Classifier'], performance_df['Precision'], marker='o', label='Precision')

    # Set plot title
    axs[row_idx, col_idx].set_title(f'Performance -- Dropped {drop_var}')

    # Customize plot appearance
    axs[row_idx, col_idx].legend()
    axs[row_idx, col_idx].grid(True)

    # Set the y-limits
    axs[row_idx, col_idx].set_ylim(0.3, 0.9)  # Adjust the values as needed

    # Remove individual x and y axis labels for each subplot
    axs[row_idx, col_idx].set_xlabel('')
    axs[row_idx, col_idx].set_ylabel('')


# Adjust spacing between subplots and main figure
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.4)

# Save the figure with higher resolution (e.g., DPI = 300)
dpi = 300  # Adjust the DPI value as needed
plt.savefig(f'ML/Performance plots/performance_subplots.png', dpi=dpi)

# Show the plot
plt.tight_layout()
plt.show()




#---------- Spatial spread of the error ------------
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)
# Create a dictionary to store the results
results = {}
# Create an additional dictionary to store instances with error > 0.5
high_error_instances = {}

# Iterate over the classifiers and their best parameters
for i, (name, clf, param_grid) in enumerate(classifiers):
    # Exclude the SVM, KNN, and Neural Network classifiers
    if name in ['SVM', 'KNN', 'Neural Network']:
        continue

    best_params_clf = best_params[name]
    # Create an instance of the classifier with the best parameters
    clf_best = clf.set_params(**best_params_clf)
    clf_best.fit(X_normalized, y)

    # Predict class probabilities on the entire dataset
    y_pred_proba = clf_best.predict_proba(X_normalized)[:, 1]

    # Calculate the absolute error for each instance
    absolute_error = np.abs(y - y_pred_proba)
    if name=='XGBoost':
        # Add the 'absolute_error' column to the data DataFrame
        data['absolute_error'] = absolute_error

    # Create a DataFrame with the geometry and absolute error
    error_df = pd.DataFrame({'geometry': data['geometry'], 'absolute_error': absolute_error})

    # Create a GeoDataFrame from the error DataFrame
    error_gdf = gpd.GeoDataFrame(error_df, geometry='geometry')

    error_gdf.to_file(f"Shapefiles/{name}_errors.shp")

    # Filter instances with error > 0.5
    high_error_instances[name] = data[absolute_error > 0.5]

    # Store the error GeoDataFrame for the current classifier
    results[name] = error_gdf

with open(f'Saved data/spatial_errors_{approach}.pkl', 'wb') as f:
    pickle.dump(results, f)
# Store the high-error instances dictionary
with open(f'Saved data/high_error_instances_{approach}.pkl', 'wb') as f:
    pickle.dump(high_error_instances, f)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Select instances with high absolute errors
high_error_data = data[results['XGBoost']['absolute_error'] > 0.3]

# Select the relevant features for analysis (exclude the 'geometry' and 'absolute_error' columns)
selected_feature_names = ['delta_time_df_primary_schools',
       'max_time_df_primary_schools', 'delta_time_df_secondary_schools',
       'max_time_df_secondary_schools', 'delta_time_df_health_centers',
       'max_time_df_health_centers', 'delta_time_df_semi_dense_urban',
       'max_time_df_semi_dense_urban', 'nearest_distance_footpath',
       'nearest_distance_bridge', 'pop_total', 'pop_ratio_max',
       'elevation_difference', 'elev_p25', 'elev_p50', 'elev_p75',
       'terrain_ruggedness']


# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(high_error_data[feature_columns])

# Perform clustering on the normalized features
kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
cluster_labels = kmeans.fit_predict(normalized_features)

# Fit PCA on the normalized features
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)
# Get the selected components (eigenvectors)
selected_components = pca.components_
# Get the names of the selected features
selected_feature_names = [feature_columns[i] for i in range(len(feature_columns)) if selected_components.T[i].any()]


plt.scatter(x=reduced_features[:, 0], y=reduced_features[:, 1], c=cluster_labels, cmap='Set2')
plt.xlabel(selected_feature_names[0])  # Replace with the name of the first feature
plt.ylabel(selected_feature_names[1])  # Replace with the name of the second feature
plt.title('Cluster Scatter Plot')
plt.show()

# Print the feature importance for each cluster
for cluster in range(kmeans.n_clusters):
    cluster_data = high_error_data[cluster_labels == cluster]
    cluster_feature_importance = cluster_data[feature_columns].mean()
    print(f"Cluster {cluster + 1} Feature Importance:")
    print(cluster_feature_importance)
    print()


# Create a new column 'cluster' to store the cluster labels
data['absolute_error'] = results['XGBoost']['absolute_error']
data['cluster'] = np.nan

# Assign cluster labels based on the absolute error threshold
data.loc[data['absolute_error'] <= 0.3, 'cluster'] = 0  # Cluster 0
data.loc[data['absolute_error'] > 0.3, 'cluster'] = 1   # Cluster 1

# Filter data points belonging to Cluster 0
cluster_0_data = data[data['cluster'] == 0]

# Filter data points belonging to Cluster 1
cluster_1_data = data[data['cluster'] == 1]

# Calculate means, medians, and percentiles for each feature in the two clusters
feature_stats = {}

for feature in selected_feature_names:
    feature_values_cluster_0 = cluster_0_data[feature]
    feature_values_cluster_1 = cluster_1_data[feature]

    # Calculate statistics for Cluster 0
    cluster_0_mean = np.mean(feature_values_cluster_0)
    cluster_0_median = np.median(feature_values_cluster_0)
    cluster_0_20th_percentile = np.percentile(feature_values_cluster_0, 20)
    cluster_0_80th_percentile = np.percentile(feature_values_cluster_0, 80)

    # Calculate statistics for Cluster 1
    cluster_1_mean = np.mean(feature_values_cluster_1)
    cluster_1_median = np.median(feature_values_cluster_1)
    cluster_1_20th_percentile = np.percentile(feature_values_cluster_1, 20)
    cluster_1_80th_percentile = np.percentile(feature_values_cluster_1, 80)

    # Store the statistics in the feature_stats dictionary
    feature_stats[feature] = {
        'Cluster 0': {
            'Mean': cluster_0_mean,
            'Median': cluster_0_median,
            '20th Percentile': cluster_0_20th_percentile,
            '80th Percentile': cluster_0_80th_percentile
        },
        'Cluster 1': {
            'Mean': cluster_1_mean,
            'Median': cluster_1_median,
            '20th Percentile': cluster_1_20th_percentile,
            '80th Percentile': cluster_1_80th_percentile
        }
    }

# Print the feature statistics
for feature, stats in feature_stats.items():
    print(f"Feature: {feature}")
    for cluster, values in stats.items():
        print(f"  {cluster}:")
        print(f"    Mean: {values['Mean']}")
        print(f"    Median: {values['Median']}")
        print(f"    20th Percentile: {values['20th Percentile']}")
        print(f"    80th Percentile: {values['80th Percentile']}")

from sklearn.preprocessing import StandardScaler

# Replace infinite values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN values
data.dropna(inplace=True)
# Normalize the features in data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(data[selected_feature_names])
data[selected_feature_names] = normalized_features

# Create a new column 'cluster' to store the cluster labels
data['cluster'] = np.nan

# Assign cluster labels based on the absolute error threshold
data.loc[(data['absolute_error'] <= 0.2) , 'cluster'] = 0  # Cluster 0
data.loc[(data['absolute_error'] > 0.5) , 'cluster'] = 1  # Cluster 1

# Filter data points belonging to Cluster 0
cluster_0_data = data[(data['cluster'] == 0) & (data['label'] == 0)]

# Filter data points belonging to Cluster 1
cluster_1_data = data[(data['cluster'] == 1) & (data['label'] == 0)]

# Calculate means, medians, and percentiles for each feature in the two clusters
feature_stats = {}

for feature in selected_feature_names:
    feature_values_cluster_0 = cluster_0_data[feature]
    feature_values_cluster_1 = cluster_1_data[feature]

    # Calculate statistics for Cluster 0
    cluster_0_mean = np.mean(feature_values_cluster_0)
    cluster_0_median = np.median(feature_values_cluster_0)
    cluster_0_20th_percentile = np.percentile(feature_values_cluster_0, 20)
    cluster_0_80th_percentile = np.percentile(feature_values_cluster_0, 80)

    # Calculate statistics for Cluster 1
    cluster_1_mean = np.mean(feature_values_cluster_1)
    cluster_1_median = np.median(feature_values_cluster_1)
    cluster_1_20th_percentile = np.percentile(feature_values_cluster_1, 20)
    cluster_1_80th_percentile = np.percentile(feature_values_cluster_1, 80)

    # Compute differences between Cluster 1 and Cluster 0
    mean_difference = abs(cluster_1_mean - cluster_0_mean)
    median_difference = abs(cluster_1_median - cluster_0_median)
    percentile_20_difference = abs(cluster_1_20th_percentile - cluster_0_20th_percentile)
    percentile_80_difference = abs(cluster_1_80th_percentile - cluster_0_80th_percentile)

    # Store the differences in the feature_stats dictionary
    feature_stats[feature] = {
        'Mean Difference': mean_difference,
        'Median Difference': median_difference,
        '20th Percentile Difference': percentile_20_difference,
        '80th Percentile Difference': percentile_80_difference
    }

# Find features with the greatest deviation in Cluster 1 from Cluster 0
max_deviation_features = []

for feature, stats in feature_stats.items():
    if (stats['Mean Difference'] > 0 or
            stats['Median Difference'] > 0 or
            stats['20th Percentile Difference'] > 0 or
            stats['80th Percentile Difference'] > 0):
        max_deviation_features.append(feature)

# Print the feature statistics
for feature, stats in feature_stats.items():
    print(f"Feature: {feature}")
    for stat, value in stats.items():
        print(f"  {stat}: {value}")

# Print features with the greatest deviation in Cluster 1 from Cluster 0
print("Features with the greatest deviation in Cluster 1 from Cluster 0:")
print(max_deviation_features)




