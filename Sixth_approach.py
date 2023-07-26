# This file incorporates the outcomes from Matthew's model: travel times
# As an additional feature to the model, I take the Delta_t for each infrastructure
# for each positive or negative label (e.g., for each bridge or non bridge site) as
# well as the greatest travel time out of the two.
# I also consider the additional features as in the previous models:
# elevation metrics (mean, percentiles, terrain ruggedness), population (total, ratio).
# Additional feature incorporated: GDP
# Additional changes to the code: population and GDP are taken from the polygon intersections with the buffer
# that make up the greatest area (if there are more than 2 polygons within the buffer)

import pandas as pd
import geopandas as gpd
import pickle
import warnings
import os
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
from shapely import wkb
import rasterio
from rasterio.mask import mask
import random
from shapely.geometry import MultiLineString

approach = "sixth"
country = "uganda"
folder = "Uganda"
approach = "sixth"

os.chdir(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}')

# ww points
ww = gpd.read_file(f'{country}_waterways_osm_shape/{country}_waterways_osm_shape.shp')
ww.crs = "EPSG:4326"
# subregion polygons
subregions = pd.read_parquet(f'travel_time/subregions.parquet')
# Read the "subregions" dataframe
subregions_gdf = gpd.GeoDataFrame(subregions)
# Decode the binary representation and create geometries
subregions_gdf['geometry'] = subregions_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
subregions_gdf = subregions_gdf.set_geometry('geometry')
subregions_gdf.crs = "EPSG:4326"
#cols_to_save = ['geometry', 'subregion_index']
#subregions_gdf[cols_to_save].to_file(f"Shapefiles/{country}_subregions.shp", driver="ESRI Shapefile")

# bridge points
bridges = pd.read_parquet(f'b2p_{country}_bridges.parquet')
# Convert bridge coordinates to pygeos Point objects
bridges_gdf = gpd.read_file(f'Shapefiles/filtered_bridge_locations.shp')
bridges_gdf.crs = "EPSG:4326"

if country=="uganda":
    # Perform a spatial join to filter waterways and bridges
    ww_filtered = gpd.sjoin(ww, subregions_gdf, how='inner', predicate='intersects')
    bridges_filtered = gpd.sjoin(bridges_gdf, subregions_gdf, how='inner', predicate='intersects')

ww_filtered[["geometry"]].to_file("Shapefiles/subregion_filtered_ww.shp", driver="ESRI Shapefile")

# travel times: merged_dfs dictionary
with open('Saved data/merged_dfs.pickle', 'rb') as file:
    merged_dfs = pickle.load(file)

projected_crs = 'EPSG:4326'  # Replace XXXX with the appropriate EPSG code
subregions_gdf = subregions_gdf.to_crs(projected_crs)
bridges_gdf = bridges_gdf.to_crs(projected_crs)

for key, df in merged_dfs.items():
    # Create columns for the time differences and maximum times
    bridges_gdf[f'delta_time_{key}'] = None
    bridges_gdf[f'max_time_{key}'] = None
    print(key)

    for index, row in bridges_gdf.iterrows():
        bridge_geometry = row.geometry
        bridge_buffer = bridge_geometry.buffer(0.001)  # Adjust the buffer distance as needed

        try:
            # Find the first polygon that intersects with the bridge
            first_intersection = subregions_gdf[subregions_gdf.intersects(bridge_buffer)].iloc[0]
        except IndexError:
            # If there is no intersection, skip to the next bridge
            continue

        # Find the nearest polygon to the bridge that is not the first intersection
        nearest_polygon = subregions_gdf[~subregions_gdf.intersects(bridge_buffer)].distance(bridge_geometry).idxmin()
        nearest_polygon = subregions_gdf.loc[nearest_polygon]

        # Save the subregion indices
        first_subregion_index = first_intersection['subregion_index']
        nearest_subregion_index = nearest_polygon['subregion_index']

        # Filter df by subregion
        filtered_schools = df[df['subregion'] == first_subregion_index]

        try:
            # Find the nearest point
            distances = filtered_schools['geometry'].distance(bridge_geometry)

            if distances.empty:
                raise ValueError('No schools found in filtered subset')

            # Get the index of the nearest point
            nearest_index = distances.idxmin()

            # Get the nearest point and its travel time
            nearest_point_1 = filtered_schools.loc[nearest_index, 'geometry']
            nearest_travel_time_1 = filtered_schools.loc[nearest_index, 'travel_time']
        except ValueError:
            nearest_travel_time_1 = np.inf

        # Filter df by subregion
        filtered_schools = df[df['subregion'] == nearest_subregion_index]

        try:
            # Find the nearest point
            distances = filtered_schools['geometry'].distance(bridge_geometry)

            if distances.empty:
                raise ValueError('No schools found in filtered subset')

            # Get the index of the nearest point
            nearest_index = distances.idxmin()

            # Get the nearest point and its travel time
            nearest_point_2 = filtered_schools.loc[nearest_index, 'geometry']
            nearest_travel_time_2 = filtered_schools.loc[nearest_index, 'travel_time']
        except ValueError:
            nearest_travel_time_2 = np.inf

        # Calculate time differences and maximum times
        bridges_gdf.at[index, f'delta_time_{key}'] = abs(nearest_travel_time_1 - nearest_travel_time_2)
        bridges_gdf.at[index, f'max_time_{key}'] = max(nearest_travel_time_1, nearest_travel_time_2)

# footpaths
footpaths = pd.read_parquet('travel_time/footpaths.parquet')
footpaths_gdf = gpd.GeoDataFrame(footpaths)
# Decode the binary representation and create geometries
footpaths_gdf['geometry'] = footpaths_gdf['geometry'].apply(lambda x: wkb.loads(x, hex=True))
footpaths_gdf = footpaths_gdf.set_geometry('geometry')

with rasterio.open("population.tif") as src_pop:
    population_raster = src_pop.read(1)
    population_raster = np.where(population_raster == -99999, 0, population_raster)  # Set no data value to NaN

with rasterio.open("Rasters/elevation.tif") as src:
    elevation_raster = src.read(1)
    elevation_transform = src.transform

radius = 16
buffer_radius = 0.0045  # Initial buffer radius (start with a smaller value)
# Define the projected CRS that you want to use for distance calculations
projected_crs = 'EPSG:4326'  

# Reproject the GeoDataFrame and bridge_geometry to the projected CRS
bridges_gdf = bridges_gdf.to_crs(projected_crs)

for index, row in bridges_gdf.iterrows():
    bridge_geometry = row['geometry']
    print(index)

    # DISTANCE TO NEAREST BRIDGE AND FOOTPATH
    # Compute distance to nearest footpath
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    distances_footpath = footpaths_gdf['geometry'].distance(bridge_geometry)
    warnings.resetwarnings()

    nearest_distance_footpath = distances_footpath.min()

    # Compute distance to nearest bridge (excluding the current bridge)
    distances_bridge = bridges_gdf[bridges_gdf.index != index]['geometry'].distance(bridge_geometry)
    nearest_distance_bridge = distances_bridge.min()

    # Update the respective columns in bridges_gdf
    bridges_gdf.at[index, 'nearest_distance_footpath'] = nearest_distance_footpath
    bridges_gdf.at[index, 'nearest_distance_bridge'] = nearest_distance_bridge


    # POPULATION COUNT
    # Create a buffer around the bridge point
    buffer_geom = bridge_geometry.buffer(buffer_radius)

    # Find the intersecting polygons
    intersecting_polygons = subregions_gdf[subregions_gdf.intersects(buffer_geom)].copy()

    # Check if there are more than two intersecting polygons
    if len(intersecting_polygons) > 2:
        # Compute intersection areas for each polygon
        intersecting_polygons['intersection_area'] = intersecting_polygons.intersection(buffer_geom).area
        # Sort the polygons by intersection area in descending order
        intersecting_polygons = intersecting_polygons.sort_values('intersection_area', ascending=False)
        # Select the first two polygons with the greatest intersection area
        intersecting_polygons = intersecting_polygons.head(2)

    try:
        # Calculate population count within the intersecting polygons
        population_count = []
        gdp_count = []

        if len(intersecting_polygons) <= 1:
            bridges_gdf.drop(index, inplace=True)
            continue
        try:

            # Iterate over the intersecting polygons
            for poly in intersecting_polygons.geometry:
                # Compute the intersection between the polygon and the buffer
                intersection_geom = poly.intersection(buffer_geom)

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
                    with rasterio.open("population.tif") as dataset:
                        # Mask the population raster using the intersection polygons
                        masked_data, _ = mask(dataset, intersection_polygons, crop=True)

                        masked_data = np.where(masked_data == -99999, 0, masked_data)

                        # Calculate the population sum within the masked area
                        population_sum = masked_data.sum()

                        # Add the population sum to the total count
                        population_count.append(population_sum)

                    with rasterio.open("Wealth/GDP2005_1km.tif") as dataset:
                        # Mask the population raster using the intersection polygons
                        masked_data, _ = mask(dataset, intersection_polygons, crop=True)

                        # Calculate the population sum within the masked area
                        gdp_mean = masked_data.mean()

                        # Add the population sum to the total count
                        gdp_count.append(gdp_mean)
        except ValueError:
            print(ValueError)
            continue

        sorted_counts = sorted(population_count, reverse=True)

        max_count = sorted_counts[0]
        second_max_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
        total_count = max_count + second_max_count

        # Get indices of the pop counts for the GDP
        max_index = population_count.index(max_count)
        second_max_index = population_count.index(second_max_count)
        max_gdp = gdp_count[max_index]
        mean_gdp = (gdp_count[max_index] + gdp_count[second_max_index])/2

        # Update the corresponding columns in bridges_df
        bridges_gdf.at[index, "pop_total"] = total_count
        bridges_gdf.at[index, "pop_ratio_max"] = max_count/total_count
        bridges_gdf.at[index, "max_gdp"] = max_gdp
        bridges_gdf.at[index, "mean_gdp"] = mean_gdp


        # ELEVATION
        # Convert bridge coordinates to pixel indices
        x_bridge, y_bridge = bridge_geometry.x, bridge_geometry.y
        # Compute the inverse of the elevation_transform
        elevation_transform_inv = ~elevation_transform
        # Transform the bridge coordinates to pixel coordinates
        pixel_coords = elevation_transform_inv * (x_bridge, y_bridge)
        x, y = int(pixel_coords[0]), int(pixel_coords[1])

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
        # Compute slope
        dx, dy = np.gradient(surrounding_elevations)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        # Compute terrain ruggedness
        terrain_ruggedness = np.std(slope)

        bridges_gdf.at[index, "elevation_difference"] = elevation_difference
        bridges_gdf.at[index, "elev_p25"] = elevation_percentiles[0]
        bridges_gdf.at[index, "elev_p50"] = elevation_percentiles[1]
        bridges_gdf.at[index, "elev_p75"] = elevation_percentiles[2]
        bridges_gdf.at[index, "terrain_ruggedness"] = terrain_ruggedness
    except IndexError as e:
        print("Caught INdexError:", e)
        continue

# Save the merged_dfs dictionary to a pickle file
with open('Saved data/positive_labels.pickle', 'wb') as file:
    pickle.dump(bridges_gdf, file)

with open('Saved data/positive_labels.pickle', 'rb') as f:
    bridges_gdf = pickle.load(f)

# ---------------- NON BRIDGE POINTS ----------------
# Set the random seed for reproducibility
random.seed(42)
# Create an empty list to store all the coordinates
all_coordinates = []

# Iterate over each geometry in 'ww'
for geom in ww['geometry']:
    # Check if the geometry is a LineString
    if isinstance(geom, LineString):
        # For single LineString, get the coordinates directly
        coords = list(geom.coords)
        # Extend the list of coordinates with the LineString coordinates
        all_coordinates.extend(coords)

# Randomly choose 3000 coordinates from all_coordinates
random_coords = random.sample(all_coordinates, 3000)
# Create a list to store the Point geometries
point_geometries = []
# Convert the random coordinates to Points
for coord in random_coords:
    point = Point(coord)
    point_geometries.append(point)

# Create a GeoDataFrame from the Point geometries
ww_gdf = gpd.GeoDataFrame(geometry=point_geometries)

from pyproj import CRS
projected_crs = CRS.from_epsg(4326)

# Step 1: Reproject the GeoDataFrames to the projected CRS
ww_gdf.crs = "EPSG:4326"
ww_gdf = ww_gdf.to_crs(projected_crs)
subregions_gdf = subregions_gdf.to_crs(projected_crs)
subregions_gdf.crs = "EPSG:4326"

for key, df in merged_dfs.items():
    # Create columns for the time differences and maximum times
    ww_gdf[f'delta_time_{key}'] = None
    ww_gdf[f'max_time_{key}'] = None
    print(key)

    for index, row in ww_gdf.iterrows():
        bridge_geometry = row.geometry
        bridge_buffer = bridge_geometry.buffer(0.001)  # Adjust the buffer distance as needed

        try:
            # Find the first polygon that intersects with the bridge
            first_intersection = subregions_gdf[subregions_gdf.intersects(bridge_buffer)].iloc[0]
            # Find the nearest polygon to the bridge that is not the first intersection
            # Step 2: Perform the distance calculations
            nearest_polygon_index = subregions_gdf[
                ~subregions_gdf.intersects(bridge_buffer)].distance(bridge_geometry).idxmin()
            nearest_polygon = subregions_gdf.iloc[nearest_polygon_index]

            # Save the subregion indices
            first_subregion_index = first_intersection['subregion_index']
            nearest_subregion_index = nearest_polygon['subregion_index']
        except IndexError:
            # Handle the case when no intersection is found
            # Drop the row from ww_gdf and continue with the next
            ww_gdf = ww_gdf.drop(index)
            continue
        
        filtered_schools = df[df['subregion'] == first_subregion_index]

        try:
            # Find the nearest point
            distances = filtered_schools['geometry'].distance(bridge_geometry)

            if distances.empty:
               continue

            # Get the index of the nearest point
            nearest_index = distances.idxmin()

            # Get the nearest point and its travel time
            nearest_point_1 = filtered_schools.loc[nearest_index, 'geometry']
            nearest_travel_time_1 = filtered_schools.loc[nearest_index, 'travel_time']
        except ValueError:
            nearest_travel_time_1 = np.inf

        # Filter df by subregion
        filtered_schools = df[df['subregion'] == nearest_subregion_index]

        try:
            # Find the nearest point
            distances = filtered_schools['geometry'].distance(bridge_geometry)

            if distances.empty:
                continue

            # Get the index of the nearest point
            nearest_index = distances.idxmin()

            # Get the nearest point and its travel time
            nearest_point_2 = filtered_schools.loc[nearest_index, 'geometry']
            nearest_travel_time_2 = filtered_schools.loc[nearest_index, 'travel_time']
        except ValueError:
            nearest_travel_time_2 = np.inf

        # Calculate time differences and maximum times
        ww_gdf.at[index, f'delta_time_{key}'] = abs(nearest_travel_time_1 - nearest_travel_time_2)
        ww_gdf.at[index, f'max_time_{key}'] = max(nearest_travel_time_1, nearest_travel_time_2)


for index, row in ww_gdf.iterrows():
    bridge_geometry = row['geometry']
    print(index)

    # DISTANCE TO NEAREST BRIDGE AND FOOTPATH
    # Compute distance to nearest footpath
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    distances_footpath = footpaths_gdf['geometry'].distance(bridge_geometry)
    warnings.resetwarnings()

    nearest_distance_footpath = distances_footpath.min()

    # Compute distance to nearest bridge (excluding the current bridge)
    distances_bridge = ww_gdf[ww_gdf.index != index]['geometry'].distance(bridge_geometry)
    nearest_distance_bridge = distances_bridge.min()

    # Update the respective columns in bridges_gdf
    ww_gdf.at[index, 'nearest_distance_footpath'] = nearest_distance_footpath
    ww_gdf.at[index, 'nearest_distance_bridge'] = nearest_distance_bridge

    # POPULATION COUNT
    # Create a buffer around the bridge point
    buffer_geom = bridge_geometry.buffer(buffer_radius)

    # Find the intersecting polygons
    intersecting_polygons = subregions_gdf[subregions_gdf.intersects(buffer_geom)].copy()

    # Check if there are more than two intersecting polygons
    if len(intersecting_polygons) > 2:
        # Compute intersection areas for each polygon
        intersecting_polygons['intersection_area'] = intersecting_polygons.intersection(buffer_geom).area
        # Sort the polygons by intersection area in descending order
        intersecting_polygons = intersecting_polygons.sort_values('intersection_area', ascending=False)
        # Select the first two polygons with the greatest intersection area
        intersecting_polygons = intersecting_polygons.head(2)

    # Calculate population count within the intersecting polygons
    population_count = []
    gdp_count = []

    if len(intersecting_polygons) <= 1:
        ww_gdf.drop(index, inplace=True)
        continue

    try:

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
                with rasterio.open("population.tif") as dataset:
                    # Mask the population raster using the intersection polygons
                    masked_data, _ = mask(dataset, intersection_polygons, crop=True)

                    masked_data = np.where(masked_data == -99999, 0, masked_data)

                    # Calculate the population sum within the masked area
                    population_sum = masked_data.sum()

                    # Add the population sum to the total count
                    population_count.append(population_sum)

                    with rasterio.open("Wealth/GDP2005_1km.tif") as dataset:
                        # Mask the population raster using the intersection polygons
                        masked_data, _ = mask(dataset, intersection_polygons, crop=True)

                        # Calculate the population sum within the masked area
                        gdp_mean = masked_data.mean()

                        # Add the population sum to the total count
                        gdp_count.append(gdp_mean)
    except ValueError:
        print(ValueError)
        continue

    sorted_counts = sorted(population_count, reverse=True)

    max_count = sorted_counts[0]
    second_max_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
    total_count = max_count + second_max_count

    # Get indices of the pop counts for the GDP
    max_index = population_count.index(max_count)
    second_max_index = population_count.index(second_max_count)
    max_gdp = gdp_count[max_index]
    mean_gdp = (gdp_count[max_index] + gdp_count[second_max_index]) / 2

    # Update the corresponding columns in bridges_df
    ww_gdf.at[index, "pop_total"] = total_count
    ww_gdf.at[index, "pop_ratio_max"] = max_count/total_count
    ww_gdf.at[index, "max_gdp"] = max_gdp
    ww_gdf.at[index, "mean_gdp"] = mean_gdp



    # ELEVATION
    # Convert bridge coordinates to pixel indices
    x_bridge, y_bridge = bridge_geometry.x, bridge_geometry.y
    # Compute the inverse of the elevation_transform
    elevation_transform_inv = ~elevation_transform
    # Transform the bridge coordinates to pixel coordinates
    pixel_coords = elevation_transform_inv * (x_bridge, y_bridge)
    x, y = int(pixel_coords[0]), int(pixel_coords[1])

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
    # Compute slope
    dx, dy = np.gradient(surrounding_elevations)
    slope = np.sqrt(dx ** 2 + dy ** 2)
    # Compute terrain ruggedness
    terrain_ruggedness = np.std(slope)

    ww_gdf.at[index, "elevation_difference"] = elevation_difference
    ww_gdf.at[index, "elev_p25"] = elevation_percentiles[0]
    ww_gdf.at[index, "elev_p50"] = elevation_percentiles[1]
    ww_gdf.at[index, "elev_p75"] = elevation_percentiles[2]
    ww_gdf.at[index, "terrain_ruggedness"] = terrain_ruggedness

# Save the negative labels to a pickle file
with open('Saved data/negative_labels.pickle', 'wb') as file:
    pickle.dump(ww_gdf, file)


# -------- MERGED -------
with open('Saved data/negative_labels.pickle', 'rb') as f:
    ww_gdf = pickle.load(f)
with open('Saved data/positive_labels.pickle', 'rb') as f:
    bridges_gdf = pickle.load(f)

bridges_gdf.drop('pop_ratio_max', axis=1, inplace=True)
ww_gdf.drop('pop_ratio_max', axis=1, inplace=True)
ww_gdf = ww_gdf.dropna()

# Add 'label' column with values 1 to bridges_gdf
bridges_gdf['label'] = 1
# Get the desired number of entries for ww_gdf (twice the size of bridges_gdf)
desired_size = 3 * len(bridges_gdf)
# Check if ww_gdf has more entries than the desired size
if len(ww_gdf) > desired_size:
    # Randomly select the desired number of rows from ww_gdf
    ww_gdf_sampled = ww_gdf.sample(n=desired_size, random_state=42)
else:
    # If ww_gdf has fewer entries, keep all of its rows
    ww_gdf_sampled = ww_gdf
ww_gdf_sampled['label'] = 0

# Merge the two GeoDataFrames based on the specified columns
merged_gdf = pd.concat([bridges_gdf, ww_gdf_sampled], ignore_index=True)
# List of columns to merge
merge_columns = [
    'geometry',
    'delta_time_df_primary_schools',
    'max_time_df_primary_schools',
    'delta_time_df_secondary_schools',
    'max_time_df_secondary_schools',
    'delta_time_df_health_centers',
    'max_time_df_health_centers',
    'delta_time_df_semi_dense_urban',
    'max_time_df_semi_dense_urban',
    'nearest_distance_footpath',
    'nearest_distance_bridge',
    'pop_total',
    'elevation_difference',
    'elev_p25',
    'elev_p50',
    'elev_p75',
    'terrain_ruggedness', 'max_gdp', 'mean_gdp',
    'label'  # Including 'label' column in merge columns
]

# Perform the merge on the specified columns
merged_gdf = merged_gdf[merge_columns]

# Save the traintest dataset to a pickle file
merged_gdf.to_pickle(f'ML/{approach} approach/train_test_data_{approach}.pickle')

