# In this file I delete the bridges that are not in/close to waterways
import os
import geopandas as gpd
import pandas as pd

country = "uganda"  # Replace "rwanda" with the desired country name, e.g., "ethiopia"
folder = "Uganda"

os.chdir(f"/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}/")

ww = gpd.read_file(f'{country}_waterways_osm_shape/{country}_waterways_osm_shape.shp')
ww.crs = "EPSG:4326"
# Load bridge locations shapefile
# bridge points
bridges = pd.read_parquet(f'b2p_{country}_bridges.parquet')
# Convert bridge coordinates to pygeos Point objects
bridges_gdf = gpd.GeoDataFrame(bridges, geometry=gpd.points_from_xy(bridges['longitude'], bridges['latitude']))
bridges_gdf.to_file(f"Shapefiles/{country}_bridges.shp", driver="ESRI Shapefile")
bridge_locations = gpd.read_file(f'Shapefiles/{country}_bridges.shp')
bridge_locations.crs = "EPSG:4326"

# Buffer distance in degrees (adjust as needed)
buffer_distance = 0.01

# Create an empty list to store the filtered bridges
filtered_bridges = []

# Iterate over each row in 'bridge_locations'
for idx, bridge in bridge_locations.iterrows():
    try:
        # Buffer the bridge location
        buffered_bridge = bridge.geometry.buffer(buffer_distance)
        # Check if the buffered bridge intersects with any Linestring in 'ww'
        if ww.intersects(buffered_bridge).any():
            # If there's intersection, add the bridge to the filtered list
            filtered_bridges.append(bridge)

    except AttributeError:
        continue

# Create a new GeoDataFrame with the filtered bridges
filtered_bridges_gdf = gpd.GeoDataFrame(filtered_bridges)
filtered_bridges_gdf.to_file(f'Shapefiles/filtered_bridge_locations.shp')

# Print the filtered GeoDataFrame
print(filtered_bridges)

