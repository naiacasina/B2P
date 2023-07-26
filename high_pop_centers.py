import numpy as np
import geopandas as gpd
import rasterio
from scipy.ndimage import label
from shapely.geometry import Point

population_raster_path = '/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/new_population.tif'

with rasterio.open(population_raster_path) as population_raster:
    population_transform = population_raster.transform

with rasterio.open(population_raster_path) as src_pop:
    population_raster = src_pop.read(1)
    population_raster = np.where(population_raster == -99999, 0, population_raster)  # Set no data value to NaN


# High density clusters
threshold = 100
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
high_population_centers_path = "/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Shapefiles/high_population_centers.shp"
high_population_gdf.to_file(high_population_centers_path)