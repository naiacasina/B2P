# This file takes raw waterways, footpaths, population and elevation as inputs
# I compute the SAME features as in the Third Approach:
#   1) population:
#                   - distance to nearest population point,
#                   - distance to the nearest high population center,
#                   - population density within a buffer centered around the bridge
#   2) elevation:
#                   - elevation difference between the bridge and the surrounding area
#                   - elevation percentiles (25th, 50th, 75th)
#   3) infrastructures:
#                   - distance to the nearest primary school
#                   - distance to the nearest secondary school
#                   - distance to the nearest health center
#                   - distance to the nearest worship center
#
# The difference is that instead of focusing on intersections as the bridge and non-bridge sites,
# I focus on any point in the waterways and do not include footpath as a feature

import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import rasterio
import random
from scipy.spatial import distance

folder = "Rwanda"
country = "rwanda"
approach = 'fourth'

# Define raster file paths
footpaths_raster_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/new_fp.tif'
waterways_raster_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/ww.tif'
population_raster_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/new_population.tif'
elevation_raster_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/new_elevation.tif'

# Open and read raster files
with rasterio.open(footpaths_raster_path) as src:
    footpaths_raster = src.read(1)

with rasterio.open(waterways_raster_path) as src:
    waterways_raster = src.read(1)
    ww_transform = src.transform

with rasterio.open(population_raster_path) as population_raster:
    population_transform = population_raster.transform
    population_width = population_raster

with rasterio.open(population_raster_path) as src_pop:
    population_raster = src_pop.read(1)
    population_raster = np.where(population_raster == -99999, 0, population_raster)  # Set no data value to NaN
    # Convert non-zero values to 1
    binary_population = np.where(population_raster > 0, 1, 0)

with rasterio.open(elevation_raster_path) as src:
    elevation_raster = src.read(1)
    elevation_transform = src.transform

# Load bridge site location
bridges_shapefile = gpd.read_file(
    f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Shapefiles/bridges_{country}.shp')

# Define features
features = ['footpaths_distance', 'waterways_distance', 'elevation_difference', 'elev_p25', 'elev_p50', 'elev_p75',
            'population_density', 'nearest_pop_center', 'nearest_high_pop_center', 'primary_distance',
            'secondary_distance', 'health_distance', 'religious_distance']

# ----- Distance transforms -----
footpaths_np = footpaths_raster.astype(bool)
waterways_np = waterways_raster.astype(bool)
population_np = binary_population.astype(bool)
# Load distance transforms
footpaths_dist_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/fp_distance_transform.npy'
waterways_dist_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/ww_distance_transform.npy'
population_dist_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/population_distance_transform.npy'
distance_transform_fp = np.load(footpaths_dist_path)
distance_transform_ww = np.load(waterways_dist_path)
distance_transform_pop = np.load(population_dist_path)

# Neighborhood Population Density
neighborhood_radius = 0.001  # Define the radius of the neighborhood around the bridge site

# Open high_population_center data and turn into list of pairs of coordinates
high_population_centers_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Shapefiles/high_population_centers.shp")
high_population_centers = list(high_population_centers_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))

# Open schools data and turn into list of pairs of coordinates
schools_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/{country}_education_facilities/education_facilities.shp")
schools = list(schools_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))
# Filter for "Primary" schools
primary_schools = list(schools_gdf[schools_gdf['school_lev'] == 'Primary'].geometry.apply(lambda geom: (geom.x, geom.y)))
# Filter for "Secondary" schools
secondary_schools = list(schools_gdf[schools_gdf['school_lev'] == 'Secondary'].geometry.apply(lambda geom: (geom.x, geom.y)))

# Open health center data and turn into list of pairs of coordinates
health_centers_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/{country}_health_facilities/health_facilities.shp")
health_centers = list(health_centers_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))

# Open religious facilities data and turn into list of pairs of coordinates
religious_facilities_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/{country}_religious_facilities/religious_facilities.shp")
religious_facilities = list(religious_facilities_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))



# Define a radius (in pixels) to consider the surrounding area of elevation
radius = 5

# ----------- Bridge sites -----------
# Extract feature values for bridge locations
bridge_data = []
coordinates_list = []  # List to store the coordinates

for index, bridge in bridges_shapefile.iterrows():
    x_bridge, y_bridge = bridge.geometry.x, bridge.geometry.y
    # Convert bridge coordinates to pixel indices
    x = int((x_bridge - population_transform[2]) / population_transform[0])
    y = int((y_bridge - population_transform[5]) / population_transform[4])
    # Append the coordinates to the list
    coordinates_list.append((x, y))

    # Footpaths
    nearest_footpath_distance = distance_transform_fp[y, x]

    # Waterways
    nearest_waterways_distance = distance_transform_ww[y, x]

    # Population
    # The nearest population unit
    nearest_pop_distance = distance_transform_pop[y, x]
    # The nearest high population distance
    distances = distance.cdist([(x_bridge, y_bridge)], high_population_centers)
    nearest_high_pop_center = np.min(distances)

    # Distance to the nearest primary school
    distances = distance.cdist([(x_bridge, y_bridge)], primary_schools)
    nearest_primary = np.min(distances)
    # Distance to the nearest secondary school
    distances = distance.cdist([(x_bridge, y_bridge)], secondary_schools)
    nearest_secondary = np.min(distances)
    # Distance to the nearest health center
    distances = distance.cdist([(x_bridge, y_bridge)], health_centers)
    nearest_health = np.min(distances)
    # Distance to the nearest religious facility
    distances = distance.cdist([(x_bridge, y_bridge)], religious_facilities)
    nearest_religious = np.min(distances)

    # Elevation
    # Elevation difference
    elevation_bridge = elevation_raster[int(y), int(x)]
    y_min = max(0, int(y - radius))
    x_min = max(0, int(x - radius))
    y_max = min(20000, int(y + radius + 1))
    x_max = min(20000, int(x + radius + 1))

    surrounding_elevations = elevation_raster[y_min:y_max, x_min:x_max]
    elevation_difference = elevation_bridge - np.mean(surrounding_elevations)
    # Elevation percentiles
    elevation_values = surrounding_elevations.flatten()
    elevation_percentiles = np.percentile(elevation_values, [25, 50, 75])

    # Population density within 1km buffer
    buffer_radius = 0.001  # Buffer radius in coordinate units
    buffer_size = int(buffer_radius / population_transform[0])  # Buffer size in pixels
    x_min, x_max = max(0, x - buffer_size), min(population_raster.shape[1], x + buffer_size + 1)
    y_min, y_max = max(0, y - buffer_size), min(population_raster.shape[0], y + buffer_size + 1)
    population_density = np.sum(population_raster[y_min:y_max, x_min:x_max])

    print(index)

    # Compute distance to the nearest high population center
    bridge_data.append(
        [nearest_footpath_distance, nearest_waterways_distance, elevation_difference, elevation_percentiles[0],
         elevation_percentiles[1], elevation_percentiles[2], population_density, nearest_pop_distance,
         nearest_high_pop_center, nearest_primary, nearest_secondary, nearest_health, nearest_religious, 1])

# Save the list to a binary file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/bridge_data_{approach}.pkl', 'wb') as file:
    pickle.dump(bridge_data, file)

# Load the list from the pickle file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/bridge_data_{approach}.pkl', 'rb') as file:
    bridge_data = pickle.load(file)

# ----------- Non-bridge sites -----------
# Get the coordinates of waterway points
waterway_points = np.where(waterways_np == 1)
waterway_coordinates = list(zip(waterway_points[1], waterway_points[0]))  # (x, y) coordinates
# Sample random points from waterways
num_samples = 3000

# Filter the coordinates that have bridges
filtered_coordinates = []
for coord in waterway_coordinates:
    x, y = coord
    if coord not in coordinates_list:
        # Your additional conditions or code here
        filtered_coordinates.append(coord)
# Sample random points from the common coordinates
random_non_bridge_points = random.sample(filtered_coordinates, num_samples)

index = 0
# Compute the feature values for the random non-bridge points
non_bridge_data = []
for x, y in random_non_bridge_points:
    # Footpaths
    nearest_footpath_distance = distance_transform_fp[y, x]

    # Waterways
    nearest_waterways_distance = distance_transform_ww[y, x]

    x_bridge = x * population_transform[0] + population_transform[2]
    y_bridge = y * population_transform[4] + population_transform[5]

    # Population
    # The nearest population unit
    nearest_pop_distance = distance_transform_pop[y, x]
    # The nearest high population distance
    distances = distance.cdist([(x_bridge, y_bridge)], high_population_centers)
    nearest_high_pop_center = np.min(distances)

    # Distance to the nearest primary school
    distances = distance.cdist([(x_bridge, y_bridge)], primary_schools)
    nearest_primary = np.min(distances)
    # Distance to the nearest secondary school
    distances = distance.cdist([(x_bridge, y_bridge)], secondary_schools)
    nearest_secondary = np.min(distances)
    # Distance to the nearest health center
    distances = distance.cdist([(x_bridge, y_bridge)], health_centers)
    nearest_health = np.min(distances)
    # Distance to the nearest religious facility
    distances = distance.cdist([(x_bridge, y_bridge)], religious_facilities)
    nearest_religious = np.min(distances)

    # Elevation
    # Elevation difference
    elevation_bridge = elevation_raster[int(y), int(x)]
    y_min = max(0, int(y - radius))
    x_min = max(0, int(x - radius))
    y_max = min(20000, int(y + radius + 1))
    x_max = min(20000, int(x + radius + 1))

    surrounding_elevations = elevation_raster[y_min:y_max, x_min:x_max]
    elevation_difference = elevation_bridge - np.mean(surrounding_elevations)
    # Elevation percentiles
    elevation_values = surrounding_elevations.flatten()
    elevation_percentiles = np.percentile(elevation_values, [25, 50, 75])

    # Population density within 1km buffer
    buffer_radius = 0.001  # Buffer radius in coordinate units
    buffer_size = int(buffer_radius / population_transform[0])  # Buffer size in pixels
    x_min, x_max = max(0, x - buffer_size), min(population_raster.shape[1], x + buffer_size + 1)
    y_min, y_max = max(0, y - buffer_size), min(population_raster.shape[0], y + buffer_size + 1)
    population_density = np.sum(population_raster[y_min:y_max, x_min:x_max])

    index = index + 1
    print(index)

    # Compute distance to the nearest high population center
    non_bridge_data.append(
        [nearest_footpath_distance, nearest_waterways_distance, elevation_difference, elevation_percentiles[0],
         elevation_percentiles[1], elevation_percentiles[2], population_density, nearest_pop_distance,
         nearest_high_pop_center, nearest_primary, nearest_secondary, nearest_health, nearest_religious, 0])

# Save the list to a binary file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/non_bridge_data_{approach}.pkl', 'wb') as file:
    pickle.dump(non_bridge_data, file)

# Load the list from the pickle file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/non_bridge_data_{approach}.pkl', 'rb') as file:
    non_bridge_data = pickle.load(file)

# Create labeled dataset
dataset = bridge_data + non_bridge_data
df = pd.DataFrame(dataset, columns=features + ['label'])

# Save dataframe
df.to_csv(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/ML/Second Approach/train_test/train_test_data_{approach}.csv',
          index=False)

# ---------- Data for prediction ---------
features_ww = ['footpaths_distance', 'waterways_distance', 'elevation_difference', 'elev_p25', 'elev_p50', 'elev_p75',
               'population_density', 'nearest_pop_center', 'nearest_high_pop_center', 'primary_distance',
            'secondary_distance', 'health_distance', 'religious_distance', 'coordinates']
# Get the coordinates of waterway points
waterway_points = np.where(waterways_np == 1)
waterway_coordinates = list(zip(waterway_points[1], waterway_points[0]))  # (x, y) coordinates

# Sample random points from waterways
num_samples = 1200  # Specify the desired number of non-bridge samples

# Plus add some original bridge data
bridge_coords = random.sample(coordinates_list, int(num_samples/4))
# Sample random points from the common coordinates
non_bridge_points = random.sample(waterway_coordinates, num_samples)

random_sampled_points = non_bridge_points + bridge_coords
# Compute the feature values for the random non-bridge points
all_waterways_data = []
coordinates_predict_list = []
index = 0
for x, y in random_sampled_points:
    # Footpaths
    nearest_footpath_distance = distance_transform_fp[y, x]

    # Waterways
    nearest_waterways_distance = distance_transform_ww[y, x]

    x_bridge = x * population_transform[0] + population_transform[2]
    y_bridge = y * population_transform[4] + population_transform[5]
    coordinates_predict_list.append((x_bridge, y_bridge))

    # Population
    # The nearest population unit
    nearest_pop_distance = distance_transform_pop[y, x]
    # The nearest high population distance
    distances = distance.cdist([(x_bridge, y_bridge)], high_population_centers)
    nearest_high_pop_center = np.min(distances)

    # Distance to the nearest primary school
    distances = distance.cdist([(x_bridge, y_bridge)], primary_schools)
    nearest_primary = np.min(distances)
    # Distance to the nearest secondary school
    distances = distance.cdist([(x_bridge, y_bridge)], secondary_schools)
    nearest_secondary = np.min(distances)
    # Distance to the nearest health center
    distances = distance.cdist([(x_bridge, y_bridge)], health_centers)
    nearest_health = np.min(distances)
    # Distance to the nearest religious facility
    distances = distance.cdist([(x_bridge, y_bridge)], religious_facilities)
    nearest_religious = np.min(distances)


    # Elevation
    # Elevation difference
    elevation_bridge = elevation_raster[int(y), int(x)]
    y_min = max(0, int(y - radius))
    x_min = max(0, int(x - radius))
    y_max = min(20000, int(y + radius + 1))
    x_max = min(20000, int(x + radius + 1))

    surrounding_elevations = elevation_raster[y_min:y_max, x_min:x_max]
    elevation_difference = elevation_bridge - np.mean(surrounding_elevations)
    # Elevation percentiles
    elevation_values = surrounding_elevations.flatten()
    elevation_percentiles = np.percentile(elevation_values, [25, 50, 75])

    # Population density within 1km buffer
    buffer_radius = 0.001  # Buffer radius in coordinate units
    buffer_size = int(buffer_radius / population_transform[0])  # Buffer size in pixels
    x_min, x_max = max(0, x - buffer_size), min(population_raster.shape[1], x + buffer_size + 1)
    y_min, y_max = max(0, y - buffer_size), min(population_raster.shape[0], y + buffer_size + 1)
    population_density = np.sum(population_raster[y_min:y_max, x_min:x_max])

    index = index + 1
    print(index)

    # Compute distance to the nearest high population center
    all_waterways_data.append(
        [nearest_footpath_distance, nearest_waterways_distance, elevation_difference, elevation_percentiles[0],
         elevation_percentiles[1], elevation_percentiles[2], population_density, nearest_pop_distance,
         nearest_high_pop_center, nearest_primary, nearest_secondary, nearest_health, nearest_religious, (x, y)])

# Save the list to a binary file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/all_waterways_data_{approach}.pkl', 'wb') as file:
    pickle.dump(all_waterways_data, file)

# Load the list from the pickle file
# with open('/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Saved data/all_waterways_data.pkl', 'rb') as file:
#    all_waterways_data = pickle.load(file)

df_predict = pd.DataFrame(all_waterways_data, columns=features_ww)

# Save dataframe
df_predict.to_csv(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/ML/Second Approach/train_test/predict_data_{approach}.csv',
                  index=False)


import pdfkit

# Path to the input HTML file
html_file = '/Users/naiacasina/Documents/SEM2/Lead D Boyd/Code/lead_peru.html'

# Path to the output PDF file
pdf_file = '/Users/naiacasina/Documents/SEM2/Lead D Boyd/Code/lead_peru2.pdf'

# Convert HTML to PDF
pdfkit.from_file(html_file, pdf_file)

