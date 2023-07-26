# This file follows the fifth approach but gives important to precision
# and/or recall, f1_score, over accuracy

import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import PartialDependenceDisplay
import geopandas as gpd
from geopandas.tools import sjoin

folder = "Uganda"
country = "uganda"
approach = 'fifth'

os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")

# Parameters
test_size = 0.2
merger = False
drop_cities = False

if merger:
    # List of countries
    countries = ["Uganda", "Rwanda"]
    approach = 'fifth'
    test_size = 0.2

    # Create an empty list to store the dataframes for each country
    dataframes = []

    # Loop over the countries
    for country in countries:
        # Set the folder and change directory
        folder = country.lower()
        os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")

        # Import train-test dataframe
        data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')

        # Add a 'country' column to the dataframe
        data['country'] = country

        # Append the dataframe to the list
        dataframes.append(data)

    # Merge all dataframes into a single dataframe
    data = pd.concat(dataframes, ignore_index=True)
else:
    # Import train-test dataframe
    data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')


if drop_cities:
    # Load shapefile for admin 1 in Uganda
    shapefile_path = "/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/Uganda/uganda_admin_boundaries/uganda_admin_1.shp"
    admin1_uganda = gpd.read_file(shapefile_path)
    kampala_admin = admin1_uganda[admin1_uganda['NAME_1'] == "Kampala"]['geometry'].iloc[0]
    data = data[data['geometry'].apply(lambda point: not point.within(kampala_admin))]


data = data.dropna()
# Assuming your DataFrame is called 'data'
X = data.drop(['label', 'geometry'], axis=1)
y = data['label']  # Target variable

# One-hot for categorical variables
categorical_features = ['cat_primary', 'cat_secondary', 'cat_health', 'cat_religious']
numerical_features = [col for col in X.columns if col not in categorical_features]

X_categorical = X[categorical_features]
X_numerical = X[numerical_features]

encoder = OneHotEncoder(sparse=False)  # Set sparse=False explicitly
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Normalize the numerical features
scaler = MinMaxScaler()
X_numerical_normalized = scaler.fit_transform(X_numerical)

# Concatenate encoded features and numerical features
X_encoded_conc = np.concatenate([X_categorical_encoded, X_numerical_normalized], axis=1)
feature_names = list(encoder.get_feature_names_out(categorical_features))
feature_names.extend(numerical_features)
X_encoded = pd.DataFrame(X_encoded_conc, columns=feature_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42)
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
# Save dictionary
# Save the performance_df dictionary to a pickle file
with open(f'Saved data/performance_{approach}_test_size_{test_size}_dropped.pkl', 'wb') as f:
    pickle.dump(performance_df, f)

# ----- Plotting feature importances -----
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
        feature_names = X_encoded.columns

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
plt.savefig(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/ML/Performance plots/feature_importances_{approach}.png')
# Show the plot
plt.show()



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

features = [0, 2, 4, 6, 8, 9, 10, 11,12,13,14,15, 16]

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




#---------- Spatial spread of the error ------------
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
    clf_best.fit(X_encoded, y)

    # Predict class probabilities on the entire dataset
    y_pred_proba = clf_best.predict_proba(X_encoded)[:, 1]

    # Calculate the absolute error for each instance
    absolute_error = np.abs(y - y_pred_proba)

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

with open(f'Saved data/data_{approach}.pkl', 'wb') as f:
    pickle.dump(data, f)

# Store the high-error instances dictionary
with open(f'Saved data/high_error_instances_{approach}.pkl', 'wb') as f:
    pickle.dump(high_error_instances, f)

with open(f'Saved data/data_{approach}.pkl', 'rb') as f:
    data = pickle.load(f)

# Analyse the errors
selected_feature_names = ['pop_tot', 'pop_ratio_max', 'footpath_distance',
       'pop_distance', 'elevation_difference', 'elev_p25', 'elev_p50',
       'elev_p75', 'terrain_ruggedness', 'cat_primary', 'cat_secondary',
       'cat_health', 'cat_religious', 'nearest_primary', 'nearest_secondary',
       'health_centers', 'religious_facilities']
# Replace infinite values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Normalize the features in data
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))  # Set the desired range between 0 and 1
# normalized_features = scaler.fit_transform(data[selected_feature_names])
# data[selected_feature_names] = normalized_features

# Create a new column 'cluster' to store the cluster labels
data['cluster'] = np.nan

# Assign cluster labels based on the absolute error threshold
data.loc[data['absolute_error'] <= 0.2, 'cluster'] = 0  # Cluster 0
data.loc[data['absolute_error'] > 0.5, 'cluster'] = 1  # Cluster 1

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


# Print the feature statistics
for feature, stats in feature_stats.items():
    print(f"Feature: {feature}")
    for stat, value in stats.items():
        print(f"  {stat}: {value}")


