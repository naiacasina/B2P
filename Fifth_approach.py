# This file takes raw waterways, footpaths, population and elevation as inputs
# I compute the SAME features as in the Fourth Approach:
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
# The difference is that, for population and infrastructure distances, I consider
# the ones in both sides of the waterways using Matthew Peterson's computed polygons for Rwanda
# and I add terrain ruggedness to the elevation features

import geopandas as gpd
import numpy as np
import pandas as pd
import os
import rasterio
import pickle
import math
from rasterio.mask import mask
import ast
from shapely.geometry import Point, MultiPolygon
from shapely.geometry import Polygon
from scipy.spatial import distance
from shapely.wkt import loads
from scipy.ndimage import sobel, generic_filter
import traceback


folder = "Uganda"
country = "uganda"
approach = 'fifth'

os.chdir(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/")

# Load positive and negative labels with shortest distances
bridges_df = pd.read_csv('Saved data/bridges_shortest_paths.csv')
ww_df = pd.read_csv('Saved data/ww_shortest_paths_Jun8.csv')


with rasterio.open('Rasters/modified_footpaths.tif') as src_footpaths:
    footpaths = src_footpaths.read(1)
    footpaths_transform = src_footpaths.transform

# Distances: compute average walking distance in km
min_latitude = -1.498
max_latitude = -1.067
avg_latitude = (min_latitude + max_latitude) / 2
degree_latitude_km = 111.32 * math.cos(math.radians(avg_latitude))
pixel_size_x = 0.0001021321699999999617
# size in km of one pixel
distance_km_x = pixel_size_x * degree_latitude_km
# take hipotenusa
h = np.sqrt((distance_km_x/2)**2+(distance_km_x/2)**2)
# don't divide by 2 because walks whole pixel (half of one, half of other)
avg_dist_pixel = (h + distance_km_x/2)

# Convert all shortest paths to distances in km
columns_to_convert = [
    "shortest_1_health",
    "shortest_2_health",
    "shortest_1_religious",
    "shortest_2_religious",
    "shortest_1_primary",
    "shortest_2_primary",
    "shortest_1_secondary",
    "shortest_2_secondary"
]

# Iterate over the columns and apply the conversion
for column in columns_to_convert:
    # Convert 999999 values to -1
    bridges_df[column] = np.where(bridges_df[column] == 999999, -1, bridges_df[column])
    ww_df[column] = np.where(ww_df[column] == 999999, -1, ww_df[column])

    # Multiply distances to kilometers by the conversion factor while ignoring -1 values
    bridges_df[column] = bridges_df[column].apply(lambda x: x * avg_dist_pixel if x != -1 else x)
    ww_df[column] = ww_df[column].apply(lambda x: x * avg_dist_pixel if x != -1 else x)

    # Convert values above 5 to -2
    bridges_df[column] = np.where(bridges_df[column] > 10, -2, bridges_df[column])
    ww_df[column] = np.where(ww_df[column] > 10, -2, ww_df[column])

    # Create the dummy variable column name
    dummy_column = "d_" + column.split("_")[1] + "_" + column.split("_")[2]

    # Set the dummy variable values based on the condition
    bridges_df[dummy_column] = np.where(bridges_df[column] > 0, 1, 0)
    ww_df[dummy_column] = np.where(ww_df[column] > 0, 1, 0)


# Load the polygons shapefile
polygons_gdf = gpd.read_file(f'Shapefiles/ww_polygons.shp')
# Assign an ID to each polygon
polygons_gdf['polygon_id'] = range(len(polygons_gdf))

# Define raster file paths
footpaths_raster_path = f'Rasters/new_fp.tif'
waterways_raster_path = f'Rasters/ww.tif'
population_raster_path = f'Rasters/new_population.tif'
elevation_raster_path = f'Rasters/new_elevation.tif'

# Open and read raster files
with rasterio.open(footpaths_raster_path) as src:
    footpaths_raster = src.read(1)
    footpaths_transform = src.transform
    footpaths_nodata = src.nodata

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


# Open high_population_center data and turn into list of pairs of coordinates
high_population_centers_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Shapefiles/high_population_centers.shp")
high_population_centers = list(high_population_centers_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))

# Open schools data and turn into list of pairs of coordinates
schools_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/{country}_education_facilities/education_facilities.shp")
schools = list(schools_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))
# Filter for "Primary" schools
#school_typ for Uganda, school_lev for Rwanda
primary_schools = list(schools_gdf[schools_gdf['school_typ'] == 'Primary School'].geometry.apply(lambda geom: (geom.x, geom.y)))
# Filter for "Secondary" schools
secondary_schools = list(schools_gdf[schools_gdf['school_typ'] == 'Secondary School'].geometry.apply(lambda geom: (geom.x, geom.y)))

# Open health center data and turn into list of pairs of coordinates
health_centers_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/{country}_health_facilities/health_facilities.shp")
health_centers = list(health_centers_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))

# Open religious facilities data and turn into list of pairs of coordinates
religious_facilities_gdf = \
    gpd.read_file(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/{country}_religious_facilities/religious_facilities.shp")
religious_facilities = list(religious_facilities_gdf.geometry.apply(lambda geom: (geom.x, geom.y)))


# Neighborhood Population Density
neighborhood_radius = 0.001  # Define the radius of the neighborhood around the bridge site
pop_radius = 0.05
# 1 pixel == 0.0001 deg (~11m). 5 pixel from each sides ~ 110m. 10 pixels ~
radius = 10

# Create empty lists to store the population counts and bridge IDs
population_counts = []
bridge_ids = []

# Convert footpaths raster to graph
footpaths_binary = np.where(footpaths_raster != footpaths_nodata, 1, 0)

def compute_terrain_ruggedness(arr):
    slope_x = sobel(arr, axis=0)
    slope_y = sobel(arr, axis=1)
    slope = np.sqrt(slope_x ** 2 + slope_y ** 2)
    ruggedness = np.mean(slope)
    return ruggedness

# ---------------- BRIDGE POINTS ---------------------
bridges_df_tt = bridges_df.copy()

for index, bridge in bridges_df_tt.iterrows():
    try:
        x_bridge, y_bridge = ast.literal_eval(bridge["bridge_point"])
        # Convert bridge coordinates to pixel indices
        x = int((x_bridge - population_transform[2]) / population_transform[0])
        y = int((y_bridge - population_transform[5]) / population_transform[4])

        # Create a buffer around the bridge point
        buffer_radius = 0.01  # Adjust the buffer radius as needed
        buffer_geom = Point(x_bridge, y_bridge).buffer(buffer_radius)

        # Find the intersecting polygons
        intersecting_polygons = polygons_gdf[polygons_gdf.intersects(buffer_geom)]

        # Calculate population count within the intersecting polygons
        population_count = []

        # Iterate over the intersecting polygons
        for poly in intersecting_polygons.geometry:
            # Compute the intersection between the polygon and the buffer
            intersection_geom = poly.intersection(buffer_geom)

            if isinstance(intersection_geom, MultiPolygon):
                if isinstance(intersection_geom, MultiPolygon):
                    # Find the polygon with the maximum area
                    largest_area = max(p.area for p in intersection_geom.geoms)
                    largest_polygon = next(p for p in intersection_geom.geoms if p.area == largest_area)
                    intersection_geom = largest_polygon

            # Check if there is a valid intersection
            if not intersection_geom.is_empty:
                # Convert the intersection geometry to a list of polygons
                intersection_polygons = [Polygon(intersection_geom)]

                # Open the raster dataset
                with rasterio.open(population_raster_path) as dataset:
                    # Mask the population raster using the intersection polygons
                    masked_data, _ = mask(dataset, intersection_polygons, crop=True)

                    masked_data = np.where(masked_data == -99999, 0, masked_data)

                    # Calculate the population sum within the masked area
                    population_sum = masked_data.sum()

                    # Add the population sum to the total count
                    population_count.append(population_sum)

        sorted_counts = sorted(population_count, reverse=True)

        max_count = sorted_counts[0]
        second_max_count = sorted_counts[1] if len(sorted_counts) > 1 else 0

        # Update the corresponding columns in bridges_df
        bridges_df_tt.at[index, "pop_count_1"] = max_count
        bridges_df_tt.at[index, "pop_count_2"] = second_max_count

        # Footpaths
        nearest_footpath_distance = distance_transform_fp[y, x]
        bridges_df_tt.at[index, "footpath_distance"] = nearest_footpath_distance
        # Population
        # The nearest population unit
        nearest_pop_distance = distance_transform_pop[y, x]
        bridges_df_tt.at[index, "pop_distance"] = nearest_pop_distance
        # The nearest high population distance
        #distances = distance.cdist([(x_bridge, y_bridge)], high_population_centers)
        #nearest_high_pop_center = np.min(distances)


        # Elevation
        # Elevation difference
        elevation_bridge = elevation_raster[int(y), int(x)]
        y_min = max(0, int(y - radius))
        x_min = max(0, int(x - radius))
        y_max = int(y + radius + 1)
        x_max = int(x + radius + 1)

        surrounding_elevations = elevation_raster[y_min:y_max, x_min:x_max]
        elevation_difference = elevation_bridge - np.mean(surrounding_elevations)
        # Elevation percentiles
        elevation_values = surrounding_elevations.flatten()
        elevation_percentiles = np.percentile(elevation_values, [25, 50, 75])
        # Compute slope
        dx, dy = np.gradient(surrounding_elevations)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        # Compute terrain ruggedness
        terrain_ruggedness = np.std(slope)

        bridges_df_tt.at[index, "elevation_difference"] = elevation_difference
        bridges_df_tt.at[index, "elev_p25"] = elevation_percentiles[0]
        bridges_df_tt.at[index, "elev_p50"] = elevation_percentiles[1]
        bridges_df_tt.at[index, "elev_p75"] = elevation_percentiles[2]
        bridges_df_tt.at[index, "terrain_ruggedness"] = terrain_ruggedness

        # Add dummies for existing infrastructure
        bridges_df_tt.at[index, "cat_primary"] = bridges_df_tt.at[index, "d_1_primary"] + bridges_df_tt.at[index, "d_2_primary"]
        bridges_df_tt.at[index, "cat_secondary"] = bridges_df_tt.at[index, "d_1_secondary"] + bridges_df_tt.at[index, "d_2_secondary"]
        bridges_df_tt.at[index, "cat_health"] = bridges_df_tt.at[index, "d_1_health"] + bridges_df_tt.at[index, "d_2_health"]
        bridges_df_tt.at[index, "cat_religious"] = bridges_df_tt.at[index, "d_1_religious"] + bridges_df_tt.at[index, "d_2_religious"]

        # Distance to the nearest primary school
        distances = distance.cdist([(x_bridge, y_bridge)], primary_schools)
        bridges_df_tt.at[index, "nearest_primary"] = np.min(distances)
        # Distance to the nearest secondary school
        distances = distance.cdist([(x_bridge, y_bridge)], secondary_schools)
        bridges_df_tt.at[index, "nearest_secondary"] = np.min(distances)
        # Distance to the nearest health center
        distances = distance.cdist([(x_bridge, y_bridge)], health_centers)
        bridges_df_tt.at[index, "health_centers"] = np.min(distances)
        # Distance to the nearest religious facility
        distances = distance.cdist([(x_bridge, y_bridge)], religious_facilities)
        bridges_df_tt.at[index, "religious_facilities"] = np.min(distances)

        print(index)
    except ValueError as e:
        print(f"Error occurred for index {index}: {e}")
        traceback.print_exc()

# Save the list to a binary file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/bridge_data_{approach}.pkl', 'wb') as file:
    pickle.dump(bridges_df_tt, file)

# Load the list from the pickle file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/bridge_data_{approach}.pkl', 'rb') as file:
    bridge_data = pickle.load(file)


# --------------- RANDOM WATERWAY POINTS ----------------
ww_df_tt = ww_df.copy()

for index, bridge in ww_df_tt.iterrows():
    point_geometry = loads(bridge["ww_point"])  # Convert WKT representation to geometry object
    x_bridge, y_bridge = point_geometry.x, point_geometry.y
    # Convert bridge coordinates to pixel indices
    x = int((x_bridge - population_transform[2]) / population_transform[0])
    y = int((y_bridge - population_transform[5]) / population_transform[4])

    # Create a buffer around the bridge point
    buffer_radius = 0.01  # Adjust the buffer radius as needed
    buffer_geom = Point(x_bridge, y_bridge).buffer(buffer_radius)

    # Find the intersecting polygons
    intersecting_polygons = polygons_gdf[polygons_gdf.intersects(buffer_geom)]

    # Calculate population count within the intersecting polygons
    population_count = []

    # Iterate over the intersecting polygons
    for poly in intersecting_polygons.geometry:
        # Compute the intersection between the polygon and the buffer
        intersection_geom = poly.intersection(buffer_geom)

        if isinstance(intersection_geom, MultiPolygon):
            if isinstance(intersection_geom, MultiPolygon):
                # Find the polygon with the maximum area
                largest_area = max(p.area for p in intersection_geom.geoms)
                largest_polygon = next(p for p in intersection_geom.geoms if p.area == largest_area)
                intersection_geom = largest_polygon

        # Check if there is a valid intersection
        if not intersection_geom.is_empty:
            # Convert the intersection geometry to a list of polygons
            intersection_polygons = [Polygon(intersection_geom)]

            # Open the raster dataset
            with rasterio.open(population_raster_path) as dataset:
                # Mask the population raster using the intersection polygons
                masked_data, _ = mask(dataset, intersection_polygons, crop=True)

                masked_data = np.where(masked_data == -99999, 0, masked_data)

                # Calculate the population sum within the masked area
                population_sum = masked_data.sum()

                # Add the population sum to the total count
                population_count.append(population_sum)

    sorted_counts = sorted(population_count, reverse=True)

    max_count = sorted_counts[0]
    second_max_count = sorted_counts[1] if len(sorted_counts) > 1 else 0

    # Update the corresponding columns in bridges_df
    ww_df_tt.at[index, "pop_count_1"] = max_count
    ww_df_tt.at[index, "pop_count_2"] = second_max_count

    # Footpaths
    nearest_footpath_distance = distance_transform_fp[y, x]
    ww_df_tt.at[index, "footpath_distance"] = nearest_footpath_distance
    # Population
    # The nearest population unit
    nearest_pop_distance = distance_transform_pop[y, x]
    ww_df_tt.at[index, "pop_distance"] = nearest_pop_distance
    # The nearest high population distance
    #distances = distance.cdist([(x_bridge, y_bridge)], high_population_centers)
    #nearest_high_pop_center = np.min(distances)


    # Elevation
    # Elevation difference
    elevation_bridge = elevation_raster[int(y), int(x)]
    y_min = max(0, int(y - radius))
    x_min = max(0, int(x - radius))
    y_max = int(y + radius + 1)
    x_max = int(x + radius + 1)

    surrounding_elevations = elevation_raster[y_min:y_max, x_min:x_max]
    elevation_difference = elevation_bridge - np.mean(surrounding_elevations)
    # Elevation percentiles
    elevation_values = surrounding_elevations.flatten()
    elevation_percentiles = np.percentile(elevation_values, [25, 50, 75])
    # Compute slope
    dx, dy = np.gradient(surrounding_elevations)
    slope = np.sqrt(dx ** 2 + dy ** 2)
    # Compute terrain ruggedness
    terrain_ruggedness = np.std(slope)

    ww_df_tt.at[index, "elevation_difference"] = elevation_difference
    ww_df_tt.at[index, "elev_p25"] = elevation_percentiles[0]
    ww_df_tt.at[index, "elev_p50"] = elevation_percentiles[1]
    ww_df_tt.at[index, "elev_p75"] = elevation_percentiles[2]
    ww_df_tt.at[index, "terrain_ruggedness"] = terrain_ruggedness

    # Add dummies for existing infrastructure
    ww_df_tt.at[index, "cat_primary"] = ww_df_tt.at[index, "d_1_primary"] + ww_df_tt.at[index, "d_2_primary"]
    ww_df_tt.at[index, "cat_secondary"] = ww_df_tt.at[index, "d_1_secondary"] + ww_df_tt.at[index, "d_2_secondary"]
    ww_df_tt.at[index, "cat_health"] = ww_df_tt.at[index, "d_1_health"] + ww_df_tt.at[index, "d_2_health"]
    ww_df_tt.at[index, "cat_religious"] = ww_df_tt.at[index, "d_1_religious"] + ww_df_tt.at[index, "d_2_religious"]

    # Distance to the nearest primary school
    distances = distance.cdist([(x_bridge, y_bridge)], primary_schools)
    ww_df_tt.at[index, "nearest_primary"] = np.min(distances)
    # Distance to the nearest secondary school
    distances = distance.cdist([(x_bridge, y_bridge)], secondary_schools)
    ww_df_tt.at[index, "nearest_secondary"] = np.min(distances)
    # Distance to the nearest health center
    distances = distance.cdist([(x_bridge, y_bridge)], health_centers)
    ww_df_tt.at[index, "health_centers"] = np.min(distances)
    # Distance to the nearest religious facility
    distances = distance.cdist([(x_bridge, y_bridge)], religious_facilities)
    ww_df_tt.at[index, "religious_facilities"] = np.min(distances)

    print(index)

# Save the list to a binary file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/ww_random_data_{approach}.pkl', 'wb') as file:
    pickle.dump(ww_df_tt, file)

# Load the list from the pickle file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/ww_random_data_{approach}.pkl', 'rb') as file:
    ww_df_tt = pickle.load(file)

# Add a "label" column to each dataframe
bridges_df_tt['label'] = 1
ww_df_tt['label'] = 0

# Build points
# Convert 'bridge_point' column to Point geometry
bridges_df_tt['geometry'] = bridges_df_tt['bridge_point'].apply(lambda point_str: Point(eval(point_str)))
# Drop the original 'bridge_point' column
bridges_df_tt.drop('bridge_point', axis=1, inplace=True)
# Convert the string representation of Point geometries to actual Point objects
ww_df_tt['geometry'] = ww_df_tt['ww_point'].apply(lambda geom_str: loads(geom_str))

# Concatenate the dataframes vertically
merged_df = pd.concat([bridges_df_tt, ww_df_tt], ignore_index=True)
merged_df['pop_tot'] = merged_df['pop_count_1'] + merged_df['pop_count_2']
merged_df['pop_ratio_max'] = np.maximum(merged_df['pop_count_1'], merged_df['pop_count_2']).div(merged_df['pop_tot'])

# Select the desired columns in merged_df
columns_to_keep = ['geometry','label','pop_tot', 'pop_ratio_max', 'footpath_distance', 'pop_distance', 'elevation_difference',
                   'elev_p25', 'elev_p50', 'elev_p75', 'terrain_ruggedness', 'cat_primary', 'cat_secondary', 'cat_health', 'cat_religious',
                   'nearest_primary', 'nearest_secondary', 'health_centers', 'religious_facilities']
merged_df_filtered = merged_df[columns_to_keep]
# Save traintest dataset
merged_df_filtered.to_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')






