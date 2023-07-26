import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import rasterio
from rasterio.transform import from_origin
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt

folder = "Uganda"
country = "uganda"
approach = 'fifth'

os.chdir(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/")

# Load the shapefile of Rwanda (or any other geospatial data of Rwanda)
shapefile_path = f'{country}_admin_boundaries/{country}_admin_0.shp'
country_map = gpd.read_file(shapefile_path)
with open(f'Saved data/spatial_errors_{approach}.pkl', 'rb') as f:
    results = pickle.load(f)

vmin = 0.0
vmax = 0.8
# Iterate over the keys in the results dictionary
for model in results.keys():
    merged_error_gdf = results[model]
    merged_error_gdf = merged_error_gdf.dropna()

    # Plot Rwanda shapefile
    country_map.plot()
    # Plot points with color-coded absolute error values
    merged_error_gdf.plot(column='absolute_error', cmap='coolwarm', legend=True, markersize=5, ax=plt.gca(),
                          vmin=vmin, vmax=vmax)    # Customize the plot
    plt.title(f'Absolute Error in Rwanda - {model}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Save the plot with high resolution
    plt.savefig(f'Figures/absolute_error_map_{model}.png', dpi=300, bbox_inches='tight')

    # Show the map plot
    plt.show()

# Create a grid over the country
xmin, ymin, xmax, ymax = country_map.total_bounds
grid_size = 0.00083  # adjust the grid size as needed
grid_x, grid_y = np.meshgrid(np.arange(xmin, xmax, grid_size), np.arange(ymax, ymin, -grid_size))

# Interpolate the absolute error values using IDW CUBIC METHOD
points = merged_error_gdf['geometry'].apply(lambda p: (p.x, p.y)).tolist()
values = merged_error_gdf['absolute_error'].tolist()
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Clip the interpolated values to the range of 0 and 1
grid_z = np.clip(grid_z, 0, 1)

# Define the output raster file path
output_raster = f'Rasters/spatial_error_{approach}.tif'

# Set up the raster metadata
transform = from_origin(xmin, ymax, grid_size, grid_size)
height, width = grid_z.shape
count = 1  # number of bands
dtype = grid_z.dtype
crs = rasterio.crs.CRS.from_epsg(4326)  # EPSG code for WGS84

# Write the interpolated surface to a raster file
with rasterio.open(output_raster, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype,
                   crs=crs, transform=transform) as dst:
    dst.write(grid_z, 1)


# Plot the interpolated surface
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(grid_x, grid_y, grid_z, cmap='coolwarm')

# Overlay the Rwanda map boundaries
rwanda_map.plot(ax=ax, color='none', edgecolor='black')

# Customize the plot appearance (title, labels, etc.)
plt.title('Interpolated Surface of Absolute Errors')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Create a colorbar
cbar = plt.colorbar(im, ax=ax, label='Absolute Error')

plt.savefig('Figures/interpolated_surface.png', dpi=300)
# Display the plot
plt.show()


# Using the kriging method from spatial stats
# Create an instance of OrdinaryKriging
# Extract x_coords and y_coords from points
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]

# Perform Ordinary Kriging
OK = OrdinaryKriging(x_coords, y_coords, values)

# Create a grid over Rwanda
xmin, ymin, xmax, ymax = rwanda_map.total_bounds
grid_size = 0.002  # adjust the grid size as needed
grid_x = np.arange(xmin, xmax, grid_size)
grid_y = np.arange(ymin, ymax, grid_size)

# Perform Ordinary Kriging
grid_z, _ = OK.execute('grid', grid_x, grid_y)

#grid_z = grid_z.reshape((len(grid_x), len(grid_y)))

# Plot the interpolated surface
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(grid_z, extent=[xmin, xmax, ymin, ymax], cmap='coolwarm')

# Overlay the Rwanda map boundaries
rwanda_map.plot(ax=ax, color='none', edgecolor='black')

# Customize the plot appearance (title, labels, etc.)
plt.title('Interpolated Surface of Absolute Errors')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Create a colorbar
cbar = plt.colorbar(im, ax=ax, label='Absolute Error')

# Display the plot
plt.show()