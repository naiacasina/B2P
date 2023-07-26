# This file follows the sixth approach using as model inputs Matthew's model's outputs as well
# but trains in one country and test in another

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
import seaborn as sns
import os
import geopandas as gpd

folder = "Rwanda"
country = "rwanda"
approach = 'sixth'

os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")

# Import train-test dataframe
data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')
data.drop('pop_ratio_max', axis=1, inplace=True)
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


# Load data from the other country
folder_2 = "Uganda"
country_2 = "uganda"
approach_2 = 'sixth'

os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder_2}/")

# Import train-test dataframe
data_2 = pd.read_pickle(f'ML/{approach_2} approach/train_test_data_{approach_2}.pickle')
data_2 = data_2.dropna()
inf_rows = data_2.isin([np.inf, -np.inf]).any(axis=1)
# Filter out rows with inf values
data_2 = data_2[~inf_rows].copy()
data_2['elev_perc_dif'] = data_2['elev_p75'] - data_2['elev_p25']

# X = data.drop(['label', 'geometry', 'max_gdp', 'mean_gdp', 'elev_p75', 'elev_p25'], axis=1)
X_2 = data_2.drop(['label', 'geometry', 'max_gdp', 'mean_gdp', 'elev_p75', 'elev_p25',
                         'delta_time_df_secondary_schools',
                        'max_time_df_primary_schools'], axis=1)
y_2 = data_2['label']  # Target variable

# Normalize the numerical features
scaler = MinMaxScaler()
X_normalized_2 = scaler.fit_transform(X_2)

#---------- Spatial spread of the error ------------
os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")
with open(f'Saved data/best_params_{approach}_test_size_{test_prop}.pkl', 'rb') as f:
    best_params = pickle.load(f)
# Create a dictionary to store the results
results = {}
# Create an additional dictionary to store instances with error > 0.5
high_error_instances = {}
# Initialize a dictionary to store performance metrics for each classifier
performance_metrics = {}

# Iterate over the classifiers and their best parameters
for i, (name, clf, param_grid) in enumerate(classifiers):
    # Exclude the SVM, KNN, and Neural Network classifiers
    if name in ['SVM', 'KNN', 'Neural Network']:
        continue

    best_params_clf = best_params[name]
    # Create an instance of the classifier with the best parameters
    clf_best = clf.set_params(**best_params_clf)
    # Train in the original country
    clf_best.fit(X_normalized, y)

    # Predict on the second country
    y_pred_proba = clf_best.predict_proba(X_normalized_2)[:, 1]
    # Predict on the second country
    y_pred = clf_best.predict(X_normalized_2)

    # Calculate performance metrics
    accuracy = accuracy_score(y_2, y_pred)
    recall = recall_score(y_2, y_pred)
    precision = precision_score(y_2, y_pred)
    # Store the metrics in the performance_metrics dictionary
    performance_metrics[name] = {'accuracy': accuracy, 'recall': recall, 'precision': precision}

    # Calculate the absolute error for each instance
    absolute_error = np.abs(y_2 - y_pred_proba)
    if name=='XGBoost':
        # Add the 'absolute_error' column to the data DataFrame
        data_2['absolute_error'] = absolute_error

    # Create a DataFrame with the geometry and absolute error
    error_df = pd.DataFrame({'geometry': data_2['geometry'], 'absolute_error': absolute_error})

    # Create a GeoDataFrame from the error DataFrame
    error_gdf = gpd.GeoDataFrame(error_df, geometry='geometry')

    error_gdf.to_file(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{country_2}/Shapefiles/{name}_errors_comp_r_u.shp")

    # Filter instances with error > 0.5
    high_error_instances[name] = data_2[absolute_error > 0.5]

    # Store the error GeoDataFrame for the current classifier
    results[name] = error_gdf

with open(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{country_2}/Saved data/comparative_spatial_errors_{approach}.pkl', 'wb') as f:
    pickle.dump(results, f)
# Store the high-error instances dictionary
with open(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{country_2}/Saved data/comparative_high_error_instances_{approach}.pkl', 'wb') as f:
    pickle.dump(high_error_instances, f)
with open(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{country_2}/Saved data/comparative_performance_{approach}.pkl', 'wb') as f:
    pickle.dump(performance_metrics, f)


# Print the performance metrics for each classifier
for name, metrics in performance_metrics.items():
    print(f"Classifier: {name}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Recall: {metrics['recall']}")
    print(f"Precision: {metrics['precision']}")
    print()


# Plot
# Extract the metrics for XGBoost and Random Forest from the performance_metrics dictionary
xgboost_metrics = performance_metrics['XGBoost']
random_forest_metrics = performance_metrics['Random Forest']

# Create a pandas DataFrame from the extracted metrics
metrics_df = pd.DataFrame({
    'Classifier': ['XGBoost', 'Random Forest'],
    'Accuracy': [xgboost_metrics['accuracy'], random_forest_metrics['accuracy']],
    'Recall': [xgboost_metrics['recall'], random_forest_metrics['recall']],
    'Precision': [xgboost_metrics['precision'], random_forest_metrics['precision']]
})

# ---- Performance plot for XGBoost and Random Forest ----
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(4, 4))

# Plot accuracy, recall, and precision
ax.plot(metrics_df['Classifier'], metrics_df['Accuracy'], marker='o', label='Accuracy')
ax.plot(metrics_df['Classifier'], metrics_df['Recall'], marker='o', label='Recall')
ax.plot(metrics_df['Classifier'], metrics_df['Precision'], marker='o', label='Precision')

# Set plot labels and title
ax.set_xlabel('Methods')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison')

# Customize plot appearance
ax.legend()
ax.grid(True)

# Set the y-limits
ax.set_ylim(0.3, 0.9)  # Adjust the values as needed

# Save the figure with higher resolution (e.g., DPI = 300)
dpi = 300  # Adjust the DPI value as needed
plt.savefig(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder_2}/ML/Performance plots/performance_plot_comparative_wrt_{country}.png', dpi=dpi)

# Show the plot
plt.show()