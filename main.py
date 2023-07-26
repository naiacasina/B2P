#
# 13rd May 2026
# ---------
# In this file I:
#   1) convert the parquet files for bridges, footpaths and waterways into raster files
#   2) compute the high_population_center list of coordinates
#   3) compute a dictionary of adjacent polygons

import pandas as pd
import shapely.wkb as wkblib
from rasterio.features import rasterize
from rasterio.crs import CRS
from shapely.geometry import mapping
import numpy as np
import geopandas as gpd
import rasterio
from scipy.ndimage import label
from shapely.geometry import Point
import pickle

country = "uganda"  # Replace "rwanda" with the desired country name, e.g., "ethiopia"
folder = "Uganda"
mypath = f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/"

# Read parquet data with OSM geometries and plot
waterways = f"Waterways/{country}_waterways_osm.parquet"
df_ww = pd.read_parquet(mypath + waterways)
footpaths = f"{country}_footpaths_osm.parquet"
df_fp = pd.read_parquet(mypath + footpaths)
bridges = f"{country}_bridges_osm.parquet"
df_b = pd.read_parquet(mypath + bridges)
ww_poly = f"subregions.parquet"
df_poly = pd.read_parquet(mypath + ww_poly)

# Turn bridges into gdf and save
geometry = [Point(x,y) for x,y in zip(df_b['longitude'], df_b['latitude'])]
gdf = gpd.GeoDataFrame(geometry=geometry)
gdf.to_file(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Shapefiles/bridges_{country}.shp')

# What is in waterways and what's the frequency
waterway_vals = df_ww["waterway"].value_counts()
print(waterway_vals)

# Waterways
geometry = df_ww["geometry"]
rivers = df_ww[df_ww['waterway'] == 'river']['geometry']
streams = df_ww[df_ww['waterway'] == 'stream']['geometry']
geom_ww = wkblib.loads(geometry)
geom_rivers = wkblib.loads(rivers)
geom_streams = wkblib.loads(streams)

# Footpaths
fp = df_fp["geometry"]
geom_fp = wkblib.loads(fp)

# Bridges
brdg = df_b["geometry"]
geom_bridges = wkblib.loads(brdg)

# Polygons
ww_poly = df_poly["geometry"]
geom_poly = wkblib.loads(ww_poly)

# Create a GeoDataFrame from the LineStrings
gdf_ww = gpd.GeoDataFrame(geometry=geom_ww)
gdf_rivers = gpd.GeoDataFrame(geometry=geom_rivers)
gdf_streams = gpd.GeoDataFrame(geometry=geom_streams)
gdf_footpaths = gpd.GeoDataFrame(geometry=geom_fp)
gdf_bridges = gpd.GeoDataFrame(geometry=geom_bridges)
gdf_poly = gpd.GeoDataFrame(geometry=geom_poly)


# Save polygons to shapefile
gdf_poly.to_file(filename=f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Shapefiles/ww_polygons.shp")

# Make a dictionary of adjacent polygons and save to pickle
adjacency_dict = {}

for index, polygon in gdf_poly.iterrows():
    fid = polygon['FID']
    adjacent_fids = []

    for other_index, other_polygon in gdf_poly.iterrows():
        if index != other_index:  # Skip the same polygon
            if polygon.geometry.touches(other_polygon.geometry):
                adjacent_fids.append(other_polygon['FID'])

    adjacency_dict[fid] = adjacent_fids

# Save the adjacency dictionary to a pickle file
with open(f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Saved data/adjacency_dict.pkl', 'wb') as file:
    pickle.dump(adjacency_dict, file)


# ----- Convert the bridges linestrings to raster -----
src_crs = CRS.from_epsg(4326)
# Determine extent of raster
minx, miny, maxx, maxy = gdf_bridges.total_bounds
resolution = 0.000102
width = (maxx - minx) / resolution
height = width

# Create new raster dataset
new_dataset = rasterio.open(
    f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/bridges.tif",
    "w",
    driver="GTiff",
    width=width,
    height=height,
    count=1,
    dtype=rasterio.uint8,
    crs=src_crs,
    transform=rasterio.Affine(resolution, 0, minx, 0, -resolution, maxy),
)

# Convert linestrings to a raster
shapes = ((mapping(line), 1) for line in gdf_bridges.geometry)
burned = rasterize(shapes=shapes, out_shape=new_dataset.shape, fill=0, transform=new_dataset.transform)

# Write raster data to file
new_dataset.write(burned, 1)
new_dataset.close()



# ----- Convert the waterways linestrings to raster -----
# Determine extent of raster
minx, miny, maxx, maxy = gdf_ww.total_bounds
width = (maxx - minx) / resolution
height = width

# Create new raster dataset
new_dataset = rasterio.open(
    f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/ww.tif",
    "w",
    driver="GTiff",
    width=width,
    height=height,
    count=1,
    dtype=rasterio.uint8,
    crs=src_crs,
    transform=rasterio.Affine(resolution, 0, minx, 0, -resolution, maxy),
)

# Convert linestrings to a raster
shapes = ((mapping(line), 1) for line in gdf_ww.geometry)
burned = rasterize(shapes=shapes, out_shape=new_dataset.shape, fill=0, transform=new_dataset.transform)

# Write raster data to file
new_dataset.write(burned, 1)
new_dataset.close()



# ----- Convert the footpaths linestrings to raster -----
# Determine extent of raster
minx, miny, maxx, maxy = gdf_footpaths.total_bounds
width = (maxx - minx) / resolution
height = width

# Create new raster dataset
new_dataset = rasterio.open(
    f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/fp.tif",
    "w",
    driver="GTiff",
    width=width,
    height=height,
    count=1,
    dtype=rasterio.uint8,
    crs=src_crs,
    transform=rasterio.Affine(resolution, 0, minx, 0, -resolution, maxy),
)

# Convert linestrings to a raster
shapes = ((mapping(line), 1) for line in gdf_footpaths.geometry)
burned = rasterize(shapes=shapes, out_shape=new_dataset.shape, fill=0, transform=new_dataset.transform)

# Write raster data to file
new_dataset.write(burned, 1)
new_dataset.close()


# -------------- Compute high population centers -----------
population_raster_path = f'/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Rasters/new_population.tif'

with rasterio.open(population_raster_path) as population_raster:
    population_transform = population_raster.transform

with rasterio.open(population_raster_path) as src_pop:
    population_raster = src_pop.read(1)
    population_raster = np.where(population_raster == -99999, 0, population_raster)  # Set no data value to NaN

# High density clusters
threshold = 30
high_population_mask = population_raster > threshold
# Perform labeling on the high population mask to identify distinct high population centers or clusters.
# This step groups connected high population pixels into clusters.
labels, num_labels = label(high_population_mask)
high_population_centers = []
# This step calculates the centroid for each labeled cluster, representing the
# approximate center of each high population center.
# iterates over the range of label IDs, which represent the different connected components in the labels array.
for label_id in range(1, num_labels +1):
    # create a boolean mask: True values are the pixels labeled with the current label_id and False values correspond to
    # other pixels.
    label_mask = labels == label_id
    label_indices = np.argwhere(label_mask)
    print(label_indices)
    centroid_y, centroid_x = label_indices.mean(axis=0)
    # Adjust centroid coordinates to match population layer extent
    centroid_x_geo, centroid_y_geo = rasterio.transform.xy(population_transform, centroid_y, centroid_x)

    high_population_centers.append((centroid_x_geo, centroid_y_geo))
    print(label_id)


# Save high_population_centers
# Create a GeoDataFrame from the high population centers
crs = src_pop.crs  # Coordinate reference system of the population raster
# Convert coordinate tuples to Point objects
high_population_centers_points = [Point(x, y) for x, y in high_population_centers]
high_population_gdf = gpd.GeoDataFrame(geometry=high_population_centers_points, crs=crs)

# Save the high population centers shapefile
high_population_centers_path = f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/Shapefiles/high_population_centers.shp"
high_population_gdf.to_file(high_population_centers_path)

