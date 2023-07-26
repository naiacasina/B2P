# Rasters.py
# 13th May 2023
# ------
# This file creates raster files out of OSM waterways and footpath data
# ------

import osmnx as ox
import rasterio.warp
from rasterio.features import rasterize
from affine import Affine

# ------------------- Waterways -------------------
# download the waterway data for the specified location from OSM
place_name = "Rwanda"
cf = '["waterway"]'
G = ox.graph_from_place(place_name, network_type="all", custom_filter=cf)

nodes, edges = ox.graph_to_gdfs(G)

# Define the desired resolution and extent of the raster
res = 0.0005
xmin, ymin, xmax, ymax = edges.total_bounds
transform = Affine(res, 0, xmin, 0, -res, ymax)

# Define the shape of the output array
width = int((xmax - xmin) / res)
height = int((ymax - ymin) / res)
out_shape = (height, width)

# Create a mask of the waterways
shapes = ((geom, 1) for geom in edges.geometry)
mask = rasterize(shapes, out_shape=out_shape, transform=transform)

# Write the mask to a raster file
with rasterio.open("/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/waterways.tif", "w", driver="GTiff", height=height, width=width, count=1, dtype=rasterio.uint8, crs=edges.crs, transform=transform) as dst:
    dst.write(mask, 1)


# ------------------- Footpaths -------------------
# download the waterway data for the specified location from OSM
place_name = "Rwanda"
cf = '["highway"~"residential|path|footway|unclassified|service|secondary|tertiary|primary|trunk|trunk_link|secondary_link|' \
     'living_street|primary_link|tertiary_link|road"]'
G_fp = ox.graph_from_place(place_name, network_type="all", custom_filter=cf)

nodes_fp, edges_fp = ox.graph_to_gdfs(G_fp)

# Define the desired resolution and extent of the raster
res = 0.0005
xmin, ymin, xmax, ymax = edges_fp.total_bounds
transform = Affine(res, 0, xmin, 0, -res, ymax)

# Define the shape of the output array
width = int((xmax - xmin) / res)
height = int((ymax - ymin) / res)
out_shape = (height, width)

# Create a mask of the waterways
shapes = ((geom, 1) for geom in edges_fp.geometry)
mask = rasterize(shapes, out_shape=out_shape, transform=transform)

# Write the mask to a raster file
with rasterio.open("/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/footpaths.tif", "w", driver="GTiff", height=height, width=width, count=1, dtype=rasterio.uint8, crs=edges_fp.crs, transform=transform) as dst:
    dst.write(mask, 1)



# ------------------- Bridges -------------------
# download the waterway data for the specified location from OSM
place_name = "Rwanda"
cf = '["bridge"]'
G_b = ox.graph_from_place(place_name, network_type="all", custom_filter=cf)

nodes_b, edges_b = ox.graph_to_gdfs(G_b)

# Define the desired resolution and extent of the raster
res = 0.0005
xmin, ymin, xmax, ymax = edges_b.total_bounds
transform = Affine(res, 0, xmin, 0, -res, ymax)

# Define the shape of the output array
width = int((xmax - xmin) / res)
height = int((ymax - ymin) / res)
out_shape = (height, width)

# Create a mask of the waterways
shapes = ((geom, 1) for geom in edges_b.geometry)
mask = rasterize(shapes, out_shape=out_shape, transform=transform)

# Write the mask to a raster file
with rasterio.open("/Users/naiacasina/Documents/SEM2/B2P/Data/Rwanda/Rasters/b.tif", "w", driver="GTiff", height=height, width=width, count=1, dtype=rasterio.uint8, crs=edges_fp.crs, transform=transform) as dst:
    dst.write(mask, 1)