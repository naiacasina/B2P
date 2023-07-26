import pickle
import os
from scipy.spatial.distance import cdist
import random
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from rasterio.features import geometry_mask
from shapely.wkt import loads

folder = "Uganda"
country = "uganda"

os.chdir(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/")

# Load the raster file
with rasterio.open('Rasters/modified_footpaths.tif') as src:
    # Convert the raster to vector format
    footpath_array = src.read(1)
    footpath_transform = src.transform

# Load the graph object
with open('Saved data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

health_centers = gpd.read_file(f'{country}_health_facilities/health_facilities.shp')
schools_gdf = gpd.read_file(f'{country}_education_facilities/education_facilities.shp')
# Create a new GeoDataFrame for primary schools
primary_gdf = schools_gdf[schools_gdf['school_typ'] == 'Primary School']
# Create a new GeoDataFrame for secondary schools
secondary_gdf = schools_gdf[schools_gdf['school_typ'] == 'Secondary School']
religious_facilities_gdf = gpd.read_file(f'{country}_religious_facilities/religious_facilities.shp')
high_pop_gdf = gpd.read_file(f'Shapefiles/high_population_centers.shp')


# Re-check that all coordinates are in the
def compute_nearest_node(graph, x, y):
    # Calculate Euclidean distance between (x, y) and each node in the graph
    distances = {
        node: ((x - int(node[0])) ** 2 + (y - int(node[1])) ** 2) ** 0.5
        for node in graph.nodes
    }
    # Find the node with the minimum distance
    nearest_node = min(distances, key=distances.get)
    return nearest_node


# ------------- Health centers -----------------
# Create an empty dataframe to store the nearest footpath coordinates
nearest_f_health = pd.DataFrame(columns=['health_point', 'footpath_coord_1'])

# Get the indices of footpath cells with value 1
indices = np.argwhere(footpath_array == 1)

for idx, center in health_centers.iterrows():
    # Get the coordinates of the health center
    center_coords = center.geometry.coords[0]
    # Convert health center coordinates to raster indices
    x_bridge = center_coords[0]
    y_bridge = center_coords[1]
    x = int((x_bridge - footpath_transform[2]) / footpath_transform[0])
    y = int((y_bridge - footpath_transform[5]) / footpath_transform[4])

    # Compute the distances between the health center and footpath cells
    distances = cdist([(y, x)], indices)  # Note the reversed order of indices

    # Find the index of the nearest footpath
    nearest_index = np.argmin(distances)

    # Get the indices of the nearest footpath
    nearest_indices = tuple(indices[nearest_index])

    if nearest_indices not in G:
        print(nearest_indices)
        nearest_indices = compute_nearest_node(G, nearest_indices[0], nearest_indices[1])
        print('Not in G')
        print(nearest_indices)

    # Store the nearest footpath coordinates
    nearest_f_health.loc[idx] = [(x_bridge, y_bridge), nearest_indices]
    print(idx)


nearest_f_health.to_csv('Saved data/health_facilities_footpaths.csv', index=False)

# ---------- Religious facilities ------------
nearest_f_religious = pd.DataFrame(columns=['religious_point', 'footpath_coord_1'])

# Get the indices of footpath cells with value 1
indices = np.argwhere(footpath_array == 1)

# Iterate over each health center
for idx, center in religious_facilities_gdf.iterrows():
    # Get the coordinates of the health center
    print(idx)
    center_coords = center.geometry.coords[0]
    # Convert health center coordinates to raster indices
    x_bridge = center_coords[0]
    y_bridge = center_coords[1]
    x = int((x_bridge - footpath_transform[2]) / footpath_transform[0])
    y = int((y_bridge - footpath_transform[5]) / footpath_transform[4])

    # Compute the distances between the health center and footpath cells
    distances = cdist([(y, x)], indices)  # Note the reversed order of indices

    # Find the index of the nearest footpath
    nearest_index = np.argmin(distances)

    # Get the indices of the nearest footpath
    nearest_indices = tuple(indices[nearest_index])

    if nearest_indices not in G:
        print(nearest_indices)
        nearest_indices = compute_nearest_node(G, nearest_indices[0], nearest_indices[1])
        print('Not in G')
        print(nearest_indices)

    # Store the nearest footpath coordinates
    nearest_f_religious.loc[idx] = [(x_bridge, y_bridge), nearest_indices]

nearest_f_religious.to_csv(f'Saved data/religious_facilities_footpaths.csv', index=False)

# ---------- Primary schools  ------------
nearest_f_primary = pd.DataFrame(columns=['primary_s_point', 'footpath_coord_1'])

# Get the indices of footpath cells with value 1
indices = np.argwhere(footpath_array == 1)

# Iterate over each health center
for idx, center in primary_gdf.iterrows():
    # Get the coordinates of the health center
    print(idx)
    center_coords = center.geometry.coords[0]
    # Convert health center coordinates to raster indices
    x_bridge = center_coords[0]
    y_bridge = center_coords[1]
    x = int((x_bridge - footpath_transform[2]) / footpath_transform[0])
    y = int((y_bridge - footpath_transform[5]) / footpath_transform[4])

    # Compute the distances between the health center and footpath cells
    distances = cdist([(y, x)], indices)  # Note the reversed order of indices

    # Find the index of the nearest footpath
    nearest_index = np.argmin(distances)

    # Get the indices of the nearest footpath
    nearest_indices = tuple(indices[nearest_index])

    if nearest_indices not in G:
        print(nearest_indices)
        nearest_indices = compute_nearest_node(G, nearest_indices[0], nearest_indices[1])
        print('Not in G')
        print(nearest_indices)

    # Store the nearest footpath coordinates
    nearest_f_primary.loc[idx] = [(x_bridge, y_bridge), nearest_indices]

nearest_f_primary.to_csv(f'Saved data/primary_schools_footpaths.csv', index=False)

# ---------- Secondary schools  ------------
nearest_f_secondary = pd.DataFrame(columns=['secondary_s_point', 'footpath_coord_1'])

# Get the indices of footpath cells with value 1
indices = np.argwhere(footpath_array == 1)

# Iterate over each health center
for idx, center in secondary_gdf.iterrows():
    # Get the coordinates of the health center
    print(idx)
    center_coords = center.geometry.coords[0]
    # Convert health center coordinates to raster indices
    x_bridge = center_coords[0]
    y_bridge = center_coords[1]
    x = int((x_bridge - footpath_transform[2]) / footpath_transform[0])
    y = int((y_bridge - footpath_transform[5]) / footpath_transform[4])

    # Compute the distances between the health center and footpath cells
    distances = cdist([(y, x)], indices)  # Note the reversed order of indices

    # Find the index of the nearest footpath
    nearest_index = np.argmin(distances)

    # Get the indices of the nearest footpath
    nearest_indices = tuple(indices[nearest_index])

    if nearest_indices not in G:
        print(nearest_indices)
        nearest_indices = compute_nearest_node(G, nearest_indices[0], nearest_indices[1])
        print('Not in G')
        print(nearest_indices)

    # Store the nearest footpath coordinates
    nearest_f_secondary.loc[idx] = [(x_bridge, y_bridge), nearest_indices]

nearest_f_secondary.to_csv(f'Saved data/secondary_school_footpaths.csv', index=False)

# ----------  High population centers  ------------
nearest_f_highpop = pd.DataFrame(columns=['highpop_c_point', 'footpath_coord_1'])

# Iterate over each health center
for idx, center in high_pop_gdf.iterrows():
    # Get the coordinates of the health center
    print(idx)
    center_coords = center.geometry.coords[0]
    # Convert health center coordinates to raster indices
    x_bridge = center_coords[0]
    y_bridge = center_coords[1]
    x = int((x_bridge - footpath_transform[2]) / footpath_transform[0])
    y = int((y_bridge - footpath_transform[5]) / footpath_transform[4])

    # Compute the distances between the health center and footpath cells
    distances = cdist([(y, x)], indices)  # Note the reversed order of indices

    # Find the index of the nearest footpath
    nearest_index = np.argmin(distances)

    # Get the indices of the nearest footpath
    nearest_indices = indices[nearest_index]

    # Convert nearest footpath indices to actual coordinates
    x_nearest = nearest_indices[1]
    y_nearest = nearest_indices[0]

    # Store the nearest footpath coordinates
    nearest_f_highpop.loc[idx] = [(x_bridge, y_bridge), (x_nearest, y_nearest)]

nearest_f_highpop.to_csv(f'Saved data/highpop_centers_footpaths.csv', index=False)


# ----------------- Waterways  -------------------
import ast
# Step 1: Read footpaths raster, bridge shapefile, and polygon shapefile
with rasterio.open('Rasters/modified_footpaths.tif') as src_footpaths:
    footpaths = src_footpaths.read(1)
    footpaths_transform = src_footpaths.transform

with rasterio.open(f'Rasters/ww.tif') as src:
    waterways_raster = src.read(1)
    ww_transform = src.transform

waterways_np = waterways_raster.astype(bool)
# Get the coordinates of waterway points
waterway_points = np.where(waterways_np == 1)
waterway_coordinates = list(zip(waterway_points[1], waterway_points[0]))  # (x, y) coordinates
random.shuffle(waterway_coordinates)
# Sample random points from waterways
num_samples = 3000

# Remove waterway rows that are near bridge sites
buffer_distance = 0.008333
# Load positive and negative labels with shortest distances
bridges_df = pd.read_csv('Saved data/nearest_bridge_coords.csv')
# Drop rows without two coordinates
bridges_df = bridges_df[bridges_df["bridge_point"].str.contains(r"\(")]
# Reset the index if desired
bridges_df = bridges_df.reset_index(drop=True)
# Create a buffer around each bridge_point coordinate
bridges_df["buffer_geometry"] = bridges_df["bridge_point"].apply(lambda x: Point(ast.literal_eval(x)).buffer(buffer_distance))

# Convert ww_point pixel coordinates to geographic coordinates
def convert_pixels_to_coords(point, transform):
    x_ww = point[0] * transform[0] + transform[2]
    y_ww = point[1] * transform[4] + transform[5]
    return (x_ww, y_ww)

# Check if ww_point coordinates fall within the buffer of any bridge_point coordinate
def is_within_buffer(ww_point_geometry, buffer_geometries):
    for buffer_geometry in buffer_geometries:
        if buffer_geometry.contains(ww_point_geometry):
            return True
    return False

# Filter the coordinates that have bridges
filtered_coords = []
previous_buffers = []  # List to store buffers of previously added waterway points
buffer_ww = 0.0083
for coord in waterway_coordinates:
    x, y = convert_pixels_to_coords(coord, footpaths_transform)
    point = Point(x,y)
    if not is_within_buffer(point, bridges_df["buffer_geometry"]):
        if not any(point.within(buffer) for buffer in previous_buffers):
            filtered_coords.append([coord, point])
            print(len(filtered_coords))
            previous_buffers.append(point.buffer(buffer_ww))
    if len(filtered_coords)>3000:
        break

df_filtered = pd.DataFrame(filtered_coords, columns=["Coordinates", "Point"])
df_filtered.to_csv('Saved data/filtered_ww_points.csv')
#filtered_coords = list(set(waterway_coordinates) - set(filtered_coordinates))

# Read polygons
polygons_gdf = gpd.read_file(f'Shapefiles/ww_polygons.shp')

# Create an empty dataframe to store the nearest coordinates
ww_df = pd.DataFrame(columns=['ww_point', 'footpath_coord_1', 'footpath_coord_2'])
# Get the indices of footpath cells with value 1
indices = np.argwhere(footpaths == 1)

df_filtered = df_filtered.drop(df_filtered.index[0])
# Step 2: Iterate over each bridge
for index, coords in df_filtered.iterrows():
    try:
        coord = coords['Coordinates']
        point = coords['Point']
        ww_point = point

        buffer_distance = 0.001  # Adjust the buffer distance as per your requirement

        buffer = ww_point.buffer(buffer_distance)

        # Compute the distances between the bridge location and footpath cells
        distances = cdist([(coord[1],coord[0])], indices)  # Note the reversed order of indices

        # Find the index of the nearest footpath
        nearest_index = np.argmin(distances)

        # Get the indices of the nearest footpath
        nearest_indices = tuple(indices[nearest_index])

        # Convert nearest indices pixels to coordinates to compute the intersection
        x_ni = nearest_indices[1] * footpaths_transform[0] + footpaths_transform[2]
        y_ni = nearest_indices[0] * footpaths_transform[4] + footpaths_transform[5]

        # if nearest_indices not in G:
        #     print(nearest_indices)
        #     nearest_indices = compute_nearest_node(G, nearest_indices[0], nearest_indices[1])
        #     print('Not in G')
        #     print(nearest_indices)
        # Check if the bridge buffer intersects with any polygons
        intersecting_polygons = polygons_gdf[polygons_gdf.intersects(buffer)]

        print(len(ww_df))
        
        if len(intersecting_polygons) <=1:
            buffer_distance = 0.0045  # Adjust the buffer distance as per your requirement
            buffer = ww_point.buffer(buffer_distance)
            intersecting_polygons = polygons_gdf[polygons_gdf.intersects(buffer)]

        if len(intersecting_polygons) > 1:
            # Find the intersecting polygon that does not contain the nearest footpath coordinate
            other_polygon = None
            for _, polygon_row in intersecting_polygons.iterrows():
                polygon_geometry = polygon_row.geometry
                if not polygon_geometry.contains(Point(x_ni,y_ni)):
                    other_polygon = polygon_row
                    break

            if other_polygon is not None:
                other_polygon_geometry = other_polygon.geometry
                # polygon_geometry = polygons_gdf[polygons_gdf['ID'] == other_polygon[0]].geometry.squeeze()

                # Read the footpaths raster file
                with rasterio.open('Rasters/modified_footpaths.tif') as src:
                    # Create a mask for the polygon geometry
                    mask_shape = [other_polygon_geometry]
                    mask = geometry_mask(mask_shape, out_shape=src.shape, transform=src.transform, invert=True)

                    # Apply the mask to the footpaths raster
                    footpaths = src.read(1, masked=True)
                    footpaths_within_polygon = footpaths * mask

                # Compute the distances between the bridge location and footpath cells in the other polygon
                other_indices = np.argwhere(footpaths_within_polygon == 1)

                distances_other = cdist([(coord[1],coord[0])], other_indices)  # Note the reversed order of indices

                try:
                    nearest_index_other = np.argmin(distances_other)
                except Exception as e:
                    print(f"Error occurred for bridge {point}: {e}")
                    continue

                if distances_other.sum() == 0:
                    nearest_index_other = 999999
                    break
                else:
                    # Find the index of the nearest footpath in the other polygon
                    nearest_index_other = np.argmin(distances_other)

                # Get the indices of the nearest footpath in the other polygon
                nearest_indices_other = tuple(other_indices[nearest_index_other])

                # if nearest_indices_other not in G:
                #     print(nearest_indices_other)
                #     nearest_indices_other = compute_nearest_node(G, nearest_indices_other[0], nearest_indices_other[1])
                #     print('Not in G')
                #     print(nearest_indices_other)

                # Store the bridge location, nearest footpath coordinates in each polygon, and other relevant information in a dataframe
                ww_df.loc[len(ww_df)] = [ww_point, nearest_indices,nearest_indices_other] # first x and then y
            else:
                # Store the bridge location and nearest footpath coordinates (without other polygon information) in a dataframe
                ww_df.loc[len(ww_df)] = [ww_point, nearest_indices, (None, None)]

    except Exception as e:
        print(f"Error occurred for bridge {point}: {e}")
        continue

        # Save result DataFrame to CSV
ww_df.to_csv('Saved data/nearest_ww_coords.csv', index=False)


# ------- DELETE INTERSECTION POINTS FOR FOOTPATHS ---------
os.chdir(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/")

# Open footpaths raster file
with rasterio.open('Rasters/new_fp.tif') as src_footpaths:
    footpaths = src_footpaths.read(1)

    # Open waterways raster file
    with rasterio.open('Rasters/new_ww.tif') as src_waterways:
        waterways = src_waterways.read(1)

        # Set footpaths pixels to zero where footpaths and waterways coincide
        footpaths[np.logical_and(footpaths == 1, waterways == 1)] = 0

        # Perform binary dilation on the waterways raster
        waterways_dilated = binary_dilation(waterways, structure=np.ones((3, 3)))

        # Perform binary erosion on the waterways raster
        waterways_eroded = binary_erosion(waterways, structure=np.ones((3, 3)))

        # Set footpaths pixels to zero where footpaths intersect with waterways or have at least two adjacent waterway pixels
        footpaths[np.logical_or(np.logical_and(footpaths == 1, waterways_dilated == 1),
                                np.logical_and(footpaths == 1, waterways_eroded == 1))] = 0

        # Create a new raster file with the modified footpaths
        with rasterio.open('Rasters/modified_footpaths.tif', 'w', **src_footpaths.profile) as dst:
            dst.write(footpaths, 1)

# ---------- BRIDGES TO FOOTPATHS -------------

# Step 1: Read footpaths raster, bridge shapefile, and polygon shapefile
with rasterio.open('Rasters/modified_footpaths.tif') as src_footpaths:
    footpaths = src_footpaths.read(1)
    footpaths_transform = src_footpaths.transform

bridge_shapefile = gpd.read_file(f'Shapefiles/bridges_{country}.shp')
bridge_shapefile['geometry'] = bridge_shapefile['geometry'].apply(lambda geom: Point(geom.x, geom.y) if geom else None)
polygons_gdf = gpd.read_file(f'Shapefiles/ww_polygons.shp')
# Get the indices of footpath cells with value 1
indices = np.argwhere(footpaths == 1)
# Create an empty dataframe to store the nearest coordinates
result_df = pd.DataFrame(columns=['bridge_point', 'footpath_coord_1', 'footpath_coord_2'])

# Step 2: Iterate over each bridge
for index, bridge_row in bridge_shapefile.iterrows():
    try:
        x_bridge, y_bridge = bridge_row.geometry.x, bridge_row.geometry.y
        bridge_point = Point(x_bridge, y_bridge)

        # Convert bridge coordinates to pixel indices
        x = int((x_bridge - footpaths_transform[2]) / footpaths_transform[0])
        y = int((y_bridge - footpaths_transform[5]) / footpaths_transform[4])

        buffer_distance = 0.001  # Adjust the buffer distance as per your requirement

        bridge_buffer = bridge_point.buffer(buffer_distance)
        
        # Compute the distances between the bridge location and footpath cells
        distances = cdist([(y, x)], indices)  # Note the reversed order of indices

        # Find the index of the nearest footpath
        nearest_index = np.argmin(distances)

        # Get the indices of the nearest footpath
        nearest_indices = tuple(indices[nearest_index])

        # Convert nearest indices pixels to coordinates to compute the intersection
        x_ni = nearest_indices[1] * footpaths_transform[0] + footpaths_transform[2]
        y_ni = nearest_indices[0] * footpaths_transform[4] + footpaths_transform[5]

        # if nearest_indices not in G:
        #     print(nearest_indices)
        #     nearest_indices = compute_nearest_node(G, nearest_indices[0], nearest_indices[1])
        #     print('Not in G')
        #     print(nearest_indices)
        # Check if the bridge buffer intersects with any polygons
        intersecting_polygons = polygons_gdf[polygons_gdf.intersects(bridge_buffer)]

        print(index)

        if len(intersecting_polygons) > 1:
            # Find the intersecting polygon that does not contain the nearest footpath coordinate
            other_polygon = None
            for _, polygon_row in intersecting_polygons.iterrows():
                polygon_geometry = polygon_row.geometry
                if not polygon_geometry.contains(Point(x_ni, y_ni)):
                    other_polygon = polygon_row
                    break

            if other_polygon is not None:
                other_polygon_geometry = other_polygon.geometry
                # polygon_geometry = polygons_gdf[polygons_gdf['ID'] == other_polygon[0]].geometry.squeeze()

                # Read the footpaths raster file
                with rasterio.open('Rasters/modified_footpaths.tif') as src:
                    # Create a mask for the polygon geometry
                    mask_shape = [other_polygon.geometry]
                    mask = geometry_mask(mask_shape, out_shape=src.shape, transform=src.transform, invert=True)

                    # Apply the mask to the footpaths raster
                    footpaths = src.read(1, masked=True)
                    footpaths_within_polygon = footpaths * mask

                # Compute the distances between the bridge location and footpath cells in the other polygon
                other_indices = np.argwhere(footpaths_within_polygon == 1)

                distances_other = cdist([(y, x)], other_indices)  # Note the reversed order of indices

                # Find the index of the nearest footpath in the other polygon
                nearest_index_other = np.argmin(distances_other)

                # Get the indices of the nearest footpath in the other polygon
                nearest_indices_other = tuple(other_indices[nearest_index_other])

                # if nearest_indices_other not in G:
                #     print(nearest_indices_other)
                #     nearest_indices_other = compute_nearest_node(G, nearest_indices_other[0], nearest_indices_other[1])
                #     print('Not in G')
                #     print(nearest_indices_other)

                # Store the bridge location, nearest footpath coordinates in each polygon, and other relevant information in a dataframe
                result_df.loc[index] = [(bridge_point.x, bridge_point.y), nearest_indices,nearest_indices_other]
            else:
                # Store the bridge location and nearest footpath coordinates (without other polygon information) in a dataframe
                result_df.loc[index] = [(bridge_point.x, bridge_point.y), nearest_indices, (None, None)]
        else:
            # Store the bridge location and nearest footpath coordinates (without other polygon information) in a dataframe
            result_df.loc[index] = [bridge_point.x, bridge_point.y, nearest_indices[0]]
    except Exception as e:
        print(f"Error occurred for bridge {index}: {e}")
        continue

        # Save result DataFrame to CSV
result_df.to_csv('Saved data/nearest_bridge_coords.csv', index=False)

# -------------- GRAPH OBJECT FOR THE FOOTPATHS ---------

# Load the footpaths raster
footpath_raster = rasterio.open(f'Rasters/modified_footpaths.tif')

with rasterio.open('Rasters/modified_footpaths.tif') as src_footpaths:
    footpaths = src_footpaths.read(1)
    footpaths_transform = src_footpaths.transform
    footpaths_crs = src_footpaths.crs


import networkx as nx


def build_graph_from_raster(raster):
    rows, cols = raster.shape

    # Create an empty graph
    graph = nx.Graph()

    # Add nodes to the graph with indices as labels
    for row in range(rows):
        for col in range(cols):
            if raster[row, col] == 1:
                graph.add_node((row, col))

    # Add edges between adjacent nodes
    for row in range(rows):
        for col in range(cols):
            if raster[row, col] == 1:
                # Check adjacent pixels
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < rows and 0 <= new_col < cols and raster[new_row, new_col] == 1:
                            graph.add_edge((row, col), (new_row, new_col))

    return graph



graph = build_graph_from_raster(footpaths)

# Save the graph object
with open(f'Saved data/graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

# Load the graph object
with open('Saved data/graph.pkl', 'rb') as f:
    G = pickle.load(f)
