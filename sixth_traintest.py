# This file follows the sixth approach using as model inputs Matthew's model's outputs as well

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

os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")

# Import train-test dataframe
data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')
data = data.dropna()
inf_rows = data.isin([np.inf, -np.inf]).any(axis=1)
# Filter out rows with inf values
data = data[~inf_rows].copy()

data['elev_perc_dif'] = data['elev_p75'] - data['elev_p25']

# Assuming your DataFrame is called 'data'
# X = data.drop(['label', 'geometry', 'max_gdp', 'mean_gdp', 'elev_p75', 'elev_p25'], axis=1)
X = data.drop(['label', 'geometry', 'max_gdp', 'mean_gdp', 'elev_p75', 'elev_p25',
                         'delta_time_df_secondary_schools',
                        'max_time_df_primary_schools'], axis=1)
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
      'min_samples_leaf': [1, 3], 'max_features': ['sqrt', 'log2']}),
    ('SVM', SVC(), {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    ('Neural Network', MLPClassifier(), {'alpha': [0.0001, 0.001, 0.01],
                                         'hidden_layer_sizes': [(64,), (128,), (64, 64)],
                                         'solver': ['adam', 'sgd'],
                                         'batch_size': [32, 64]})
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
with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped.pkl', 'wb') as f:
    pickle.dump(performance_df, f)

with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'wb') as f:
    pickle.dump(best_params, f)

# ----- Partial dependence plots -----
#Train above with the following:
classifiers = [
    ('XGBoost', xgb.XGBClassifier(),
     {'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5], 'max_depth': [1], 'learning_rate': [0.1, 0.01],
      'subsample': [0.8, 1.0], 'gamma': [0, 0.1, 0.2]})
]

# Assuming X_train and y_train are the training data
feature_names = X.columns.to_list()

best_params_clf = best_params['XGBoost']
clf = xgb.XGBClassifier()
# Create an instance of the classifier with the best parameters
clf_best = clf.set_params(**best_params_clf)
# Fit the XGBoost classifier with max_depth=1
clf_best.fit(X_train, y_train)

features = [0, 2, 4, 6, 8, 9, 10, 16]

# Compute partial dependence plots
display = PartialDependenceDisplay.from_estimator(clf_best, X_train, features, feature_names=feature_names)
# Customize plot appearance
fig, ax = plt.subplots(figsize=(14, 6))  # Set the figure size
display.plot(ax=ax, n_cols=4)  # Set the number of columns for subplots

# Add a title
ax.set_title('Partial dependence plots', fontsize=16)
# Modify x-axis label size
ax.set_xlabel('Feature Values', fontsize=8)
# Remove y-axis labels
ax.set_ylabel('')
# Make x-axis ticks smaller
ax.tick_params(axis='x', labelsize=6)
# Add more separation between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.75)

# Save the plot with good resolution
plt.savefig(f'ML/Partial plots/partial_dependence_plots_{approach}.png', dpi=300)
# Show the plot
plt.show()


# ----- Plotting feature importances -----
with open(f'Saved data/performance_{approach}_test_size_{test_prop}_dropped.pkl', 'rb') as f:
    performance_df = pickle.load(f)
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)
# Set a color palette
colors = sns.color_palette('pastel')

# Create a figure and subplots
fig, axs = plt.subplots(len(classifiers) - 3, 1, figsize=(8, (len(classifiers) - 3) * 4))

# Iterate over the classifiers
for i, (name, clf, param_grid) in enumerate(classifiers):
    # Exclude the SVM, KNN, and Neural Network classifiers
    if name in ['SVM', 'KNN', 'Neural Network']:
        continue

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
        axs[i].set_xlabel('Features' if i == len(classifiers) - 4 else '')
        axs[i].set_ylabel('Importance')
        axs[i].set_title(f'Feature Importances - {name}')
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].grid(axis='y', linestyle='--', alpha=0.5)

        # Omit x labels in the top subplot
        if i != len(classifiers) - 4:
            axs[i].set_xticklabels([])
    except AttributeError:
        # Skip classifiers without feature importances
        continue

# Adjust spacing between subplots
plt.tight_layout()
# Save the figure
plt.savefig(f'ML/Performance plots/feature_importances_{approach}_test_size_{test_prop}.png')
# Show the plot
plt.show()


# --------- Leaving one out for the training and testing ---------
from sklearn.metrics import accuracy_score, precision_score, recall_score

with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)

# Train XGBoost classifier with all features
xgb_clf = xgb.XGBClassifier(**best_params['XGBoost'])
xgb_clf.fit(X_train, y_train)

# Make predictions on the test set using the trained model
y_pred_all_features = xgb_clf.predict(X_test)
accuracy_all_features = accuracy_score(y_test, y_pred_all_features)
precision_all_features = precision_score(y_test, y_pred_all_features)
recall_all_features = recall_score(y_test, y_pred_all_features)

# Initialize the XGBoost classifier with the best parameters
best_xgb = xgb.XGBClassifier(**best_params['XGBoost'])

# Create empty dictionaries to store the metrics for each feature
accuracy_scores = {}
precision_scores = {}
recall_scores = {}

# Iterate over each feature and evaluate the model after dropping it
for feature in X.columns:
    # Drop the current feature from the dataset
    X_dropped = X.drop(feature, axis=1)

    # Normalize the numerical features
    X_dropped_normalized = scaler.fit_transform(X_dropped)

    # Split the data into training and testing sets
    X_train_dropped, X_test_dropped, y_train, y_test = train_test_split(
        X_dropped_normalized, y, test_size=test_prop, random_state=42
    )

    # Fit the XGBoost classifier on the training data
    best_xgb.fit(X_train_dropped, y_train)

    # Make predictions on the test set
    y_pred = best_xgb.predict(X_test_dropped)

    # Calculate and store the metrics
    accuracy_scores[feature] = accuracy_score(y_test, y_pred)
    precision_scores[feature] = precision_score(y_test, y_pred)
    recall_scores[feature] = recall_score(y_test, y_pred)

# Convert the metric dictionaries into dataframes
accuracy_df = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy'])
precision_df = pd.DataFrame.from_dict(precision_scores, orient='index', columns=['Precision'])
recall_df = pd.DataFrame.from_dict(recall_scores, orient='index', columns=['Recall'])

# Plot the metrics for each dropped feature
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Set the minimum y-range to 0.5
y_min = 0.7
y_max = 0.9
accuracy_df.plot(kind='bar', ax=axes[0], legend=False)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('XGBoost Performance after Dropping Features')
axes[0].set_ylim(y_min, y_max)  # Set the y-range for accuracy plot

precision_df.plot(kind='bar', ax=axes[1], legend=False)
axes[1].set_ylabel('Precision')
axes[1].set_ylim(y_min, y_max)  # Set the y-range for precision plot

recall_df.plot(kind='bar', ax=axes[2], legend=False)
axes[2].set_ylabel('Recall')
axes[2].set_ylim(0.5, 0.7)  # Set the y-range for recall plot

for ax, metric in zip(axes, ['Accuracy', 'Precision', 'Recall']):
    all_features_value = accuracy_all_features if metric == 'Accuracy' else \
                         precision_all_features if metric == 'Precision' else \
                         recall_all_features
    ax.axhline(y=all_features_value, color='green', linestyle='dashed', label='All Features')


# Hide x-axis labels on the first two plots
axes[0].xaxis.set_ticklabels([])
axes[1].xaxis.set_ticklabels([])

# Set x-axis labels on the third plot
axes[2].set_xlabel('Dropped Feature')

plt.xlabel('Dropped Feature')

plt.tight_layout()
plt.savefig(f'ML/Performance plots/feature_drop_{approach}.png', dpi=300, bbox_inches='tight')
plt.show()



# Sort the dataframes in descending order by index (y-axis features)
accuracy_df.sort_index(ascending=False, inplace=True)
precision_df.sort_index(ascending=False, inplace=True)
recall_df.sort_index(ascending=False, inplace=True)


# Plot the metrics for each dropped feature
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Set the minimum y-range to 0.5
y_min = 0.7
y_max = 0.9

# Plot the accuracy values
accuracy_df.plot(kind='barh', ax=axes[0], legend=False)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('XGBoost Performance after Dropping Features')
axes[0].set_xlim(y_min, y_max)  # Set the x-range for accuracy plot

# Plot the precision values
precision_df.plot(kind='barh', ax=axes[1], legend=False)
axes[1].set_xlabel('Precision')
axes[1].set_xlim(y_min, y_max)  # Set the x-range for precision plot

# Plot the recall values
recall_df.plot(kind='barh', ax=axes[2], legend=False)
axes[2].set_xlabel('Recall')
axes[2].set_xlim(0.5, 0.7)  # Set the x-range for recall plot

# Add vertical lines for the all-feature values
for ax, metric in zip(axes, ['Accuracy', 'Precision', 'Recall']):
    all_features_value = accuracy_all_features if metric == 'Accuracy' else \
                         precision_all_features if metric == 'Precision' else \
                         recall_all_features
    ax.axvline(x=all_features_value, color='green', linestyle='dashed', label='All Features')

# Hide y-axis labels on the first two plots
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])

# Set y-axis labels on the third plot
axes[0].set_ylabel('Dropped Feature')

# Adjust spacing
plt.tight_layout()

# Save the plot with good quality
plt.savefig(f'ML/Performance plots/feature_drop_{approach}.png', dpi=300, bbox_inches='tight')

# Show the plot
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




