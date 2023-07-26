# This file merges:
#   a) travel time data
#   b) footpaths
#   c) subregions
# for Uganda

import os
import pandas as pd
from shapely import wkb
import geopandas as gpd

country = "uganda"
os.chdir("/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/Uganda/")


# -------------- MERGE TRAVEL TIMES ------------------
# Define the folder path
folder_path = "travel_time/"
type = 'paths'

# Initialize empty lists to store the dataframes for each category
primary_dfs = []
secondary_dfs = []
health_dfs = []
semi_dense_urban_dfs = []

# Iterate through each subfolder in the travel_time folder
for root, dirs, files in os.walk(folder_path):
    for directory in dirs:
        subfolder_path = os.path.join(root, directory)

        # Check if the subfolder contains the model_outputs folder
        model_outputs_path = os.path.join(subfolder_path, "model_outputs")
        if not os.path.exists(model_outputs_path):
            continue

        # Read the desired .parquet files in the model_outputs folder
        file_paths = [
            os.path.join(model_outputs_path, file_name)
            for file_name in [
                f"travel_{type}_to_primary_schools_optimal.parquet",
                f"travel_{type}_to_secondary_schools_optimal.parquet",
                f"travel_{type}_to_health_centers_optimal.parquet",
                f"travel_{type}_to_semi_dense_urban_optimal.parquet"
            ]
        ]

        dfs = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                dfs.append(df)

        # Add the dataframes to the corresponding category lists
        if len(dfs) == 4:
            primary_dfs.append(dfs[0])
            secondary_dfs.append(dfs[1])
            health_dfs.append(dfs[2])
            semi_dense_urban_dfs.append(dfs[3])

# Merge the dataframes within each category
merged_primary_df = pd.concat(primary_dfs)
merged_secondary_df = pd.concat(secondary_dfs)
merged_health_df = pd.concat(health_dfs)
merged_semi_dense_urban_df = pd.concat(semi_dense_urban_dfs)

# Save merged dfs to parquet files
merged_primary_df.to_parquet(f"travel_time/travel_{type}_to_primary_schools_optimal.parquet", index=False)
merged_secondary_df.to_parquet(f"travel_time/travel_{type}_to_secondary_schools_optimal.parquet", index=False)
merged_health_df.to_parquet(f"travel_time/travel_{type}_to_health_centers_optimal.parquet", index=False)
merged_semi_dense_urban_df.to_parquet(f"travel_time/travel_{type}_to_semi_dense_urban_optimal.parquet", index=False)



# -------------- MERGE SUBREGIONS ------------------
# Define the folder path
folder_path = "travel_time/"
type = 'subregions'

# Initialize empty lists to store the dataframes for each category
subregions_dfs = []
footpaths_dfs = []

# Iterate through each subfolder in the travel_time folder
for root, dirs, files in os.walk(folder_path):
    for directory in dirs:
        subfolder_path = os.path.join(root, directory)

        # Check if the subfolder contains the model_outputs folder
        model_outputs_path = os.path.join(subfolder_path, "subregions")
        if not os.path.exists(model_outputs_path):
            continue

        # Read the desired .parquet files in the model_outputs folder
        file_paths = [
            os.path.join(model_outputs_path, file_name)
            for file_name in [
                f"subregions.parquet",
                f"footpaths.parquet"
            ]
        ]

        dfs = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                dfs.append(df)

        # Add the dataframes to the corresponding category lists
        if len(dfs) == 2:
            subregions_dfs.append(dfs[0])
            footpaths_dfs.append(dfs[1])

# Merge the dataframes within each category
merged_subregions_df = pd.concat(subregions_dfs)
merged_footpaths_df = pd.concat(footpaths_dfs)

# Save merged dfs to parquet files
merged_footpaths_df.to_parquet(f"travel_time/footpaths.parquet", index=False)
merged_subregions_df.to_parquet(f"travel_time/subregions.parquet", index=False)


# -------------- MERGE WATERWAYS ------------------
import os
import pandas as pd

travel_time_dir = "travel_time/"  # Replace with the actual path to your "travel_time" directory

# Create an empty list to store the DataFrames
dfs = []

# Loop through the folders in the "travel_time" directory and read the "folder_name_waterways_osm.parquet" files
for folder_name in os.listdir(travel_time_dir):
    folder_path = os.path.join(travel_time_dir, folder_name)
    file_name = f"{folder_name}_waterways_osm.parquet"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        dfs.append(df)

# Merge the DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a file inside the "travel_time" folder
output_file = os.path.join(travel_time_dir, f"{country}_waterways_osm.parquet")
merged_df.to_parquet(output_file, index=False)

# Create shapefile
merged_df['geometry'] = merged_df['geometry'].apply(lambda x: wkb.loads(x, hex=True))
merged_df['timestamp'] = merged_df['timestamp'].astype(str)
gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')
# Save shapefile
output_shapefile = f"{country}_waterways_osm_shape/{country}_waterways_osm_shape.shp"
gdf.to_file(output_shapefile)