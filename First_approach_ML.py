# This file takes raw waterways, footpaths, population and elevation as input
# and bridge site location as output
# To compute non-bridge sites, I take random points within the boundaries of
# Rwanda
# I train a Random Forest model considering class imbalance

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Define raster file paths
footpaths_raster_path = '/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/new_fp.tif'
waterways_raster_path = '/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/ww.tif'
population_raster_path = '/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/new_population.tif'
elevation_raster_path = '/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/new_elevation.tif'

# Open and read raster files
with rasterio.open(footpaths_raster_path) as src:
    footpaths_raster = src.read(1)

with rasterio.open(waterways_raster_path) as src:
    waterways_raster = src.read(1)

with rasterio.open(population_raster_path) as src:
    population_raster = src.read(1)
    population_raster = np.where(population_raster == -99999, 0, population_raster)  # Set no data value to NaN

with rasterio.open(elevation_raster_path) as src:
    elevation_raster = src.read(1)

# Load shapefile data
boundaries_shapefile = gpd.read_file('/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/rwanda_admin_boundaries/rwanda_admin_0.shp')
bridges_shapefile = gpd.read_file('/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Shapefiles/bridge_locations.shp')

# Define features
features = ['footpaths_density', 'waterways_density', 'population_count', 'elevation']

# Extract feature values for bridge locations
bridge_data = []
for index, bridge in bridges_shapefile.iterrows():
    x, y = int(bridge.geometry.x), int(bridge.geometry.y)
    footpaths_value = footpaths_raster[y, x]
    waterways_value = waterways_raster[y, x]
    population_value = population_raster[y, x]
    elevation_value = elevation_raster[y, x]
    bridge_data.append([footpaths_value, waterways_value, population_value, elevation_value, 1])

# Randomly sample non-bridge locations within the country
non_bridge_data = []
num_samples = len(bridge_data)  # Adjust the number of samples if desired

while len(non_bridge_data) < num_samples:
    x = np.random.uniform(boundaries_shapefile.bounds.minx, boundaries_shapefile.bounds.maxx)
    y = np.random.uniform(boundaries_shapefile.bounds.miny, boundaries_shapefile.bounds.maxy)
    point = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))
    non_bridge = point[~point.intersects(bridges_shapefile.unary_union)].iloc[0]

    x, y = int(non_bridge.geometry.x), int(non_bridge.geometry.y)
    footpaths_value = footpaths_raster[y, x]
    waterways_value = waterways_raster[y, x]
    population_value = population_raster[y, x]
    elevation_value = elevation_raster[y, x]
    non_bridge_data.append([footpaths_value, waterways_value, population_value, elevation_value, 0])


# Create labeled dataset
dataset = bridge_data + non_bridge_data
df = pd.DataFrame(dataset, columns=features + ['label'])

# Split dataset into training and testing sets
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
oversampler = SMOTE(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# Train Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Predict bridge locations for the entire region
all_data = []
for index, boundary in boundaries_shapefile.iterrows():
    x, y = boundary.geometry.centroid.x, boundary.geometry.centroid.y
    footpaths_value = footpaths_raster[y, x]
    waterways_value = waterways_raster[y, x]
    population_value = population_raster[y, x]
    elevation_value = elevation_raster[y, x]
    all_data.append([footpaths_value, waterways_value, population_value, elevation_value])

all_data = np.array(all_data)
bridge_predictions = clf.predict(all_data)

# Add the predicted labels to the boundaries shapefile
boundaries_shapefile['bridge_predicted'] = bridge_predictions

# Save the updated shapefile
boundaries_shapefile.to_file('predicted_bridges.shp')


# The low precision, recall, and F1-score, as well as the relatively low accuracy,
# indicate that the current set of features may not be sufficient for accurately
# predicting bridge locations. Including additional relevant features, such as distance
# to the nearest population node, density of population nodes, and distance to the nearest footpath,
# could indeed help improve the results. These additional features can provide more contextual
# information and capture important spatial relationships.
