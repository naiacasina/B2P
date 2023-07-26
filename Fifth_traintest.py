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
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder

folder = "Rwanda"
country = "rwanda"
approach = 'fifth'

os.chdir(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/")

with rasterio.open(
        f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/new_population.tif') as population_raster:
    population_transform = population_raster.transform

# Import train-test dataframe
data = pd.read_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')
data = data.dropna()
# Assuming your DataFrame is called 'data'
X = data.drop(['label'], axis=1)
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
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# Define classifiers with their respective hyperparameter spaces for grid search
classifiers = [
    ('XGBoost', xgb.XGBClassifier(),
     {'reg_alpha': [0, 0.1, 0.5], 'reg_lambda': [0, 0.1, 0.5], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01]}),
    ('Random Forest', RandomForestClassifier(),
     {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}),
    ('SVM', SVC(), {'C': [1, 10, 100]}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    ('Neural Network', MLPClassifier(), {'alpha': [0.0001, 0.001, 0.01]})
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
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
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
with open(f'Saved data/performance_{approach}.pkl', 'wb') as f:
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
plt.savefig(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/ML/Performance plots/feature_importances_{approach}.png')
# Show the plot
plt.show()


#---------- Predictions ------------



