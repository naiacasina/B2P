import pandas as pd
import os
import pickle
import geopandas as gpd
from shapely.geometry import Point

folder = "Uganda"
os.chdir(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/{folder}')

# Read the parquet file
df_travel_primary = pd.read_parquet(f'travel_time/travel_time_to_primary_schools_optimal.parquet')
df_travel_secondary = pd.read_parquet(f'travel_time/travel_time_to_secondary_schools_optimal.parquet')
df_travel_health = pd.read_parquet(f'travel_time/travel_time_to_health_centers_optimal.parquet')
df_travel_urban = pd.read_parquet(f'travel_time/travel_time_to_semi_dense_urban_optimal.parquet')

df_paths_primary = pd.read_parquet(f'travel_time/travel_paths_to_primary_schools_optimal.parquet')
df_paths_secondary = pd.read_parquet(f'travel_time/travel_paths_to_secondary_schools_optimal.parquet')
df_paths_health = pd.read_parquet(f'travel_time/travel_paths_to_health_centers_optimal.parquet')
df_paths_urban = pd.read_parquet(f'travel_time/travel_paths_to_semi_dense_urban_optimal.parquet')

# Create a dictionary to store the merged dataframes
merged_dfs = {}

# Define the list of prefixes for dataframe names
prefixes = ['primary_schools', 'secondary_schools', 'health_centers', 'semi_dense_urban']

# Iterate over the prefixes
for prefix in prefixes:
    # Read the travel time dataframe
    df_travel = pd.read_parquet(f'travel_time/travel_time_to_{prefix}_optimal.parquet')
    # Read the travel paths dataframe
    df_paths = pd.read_parquet(f'travel_time/travel_paths_to_{prefix}_optimal.parquet')

    # Merge the dataframes on 'row' and 'col' columns
    merged_df = df_travel.merge(df_paths, on=['row', 'col'])

    # Select the desired columns
    desired_columns = ['x', 'y', 'travel_time', 'destination_coords', 'subregion']
    merged_df = merged_df[desired_columns]

    # Assign the merged dataframe to the corresponding key in the dictionary
    merged_dfs[f'df_{prefix}'] = merged_df

    print(prefix)

for df_name, df in merged_dfs.items():
    # Create the 'geometry' column
    gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df['x'], df['y'])])
    # Update the merged_dfs dictionary with the modified GeoDataFrame
    merged_dfs[df_name] = gdf
    print(df_name)

# Save the merged_dfs dictionary to a pickle file
with open('Saved data/merged_dfs.pickle', 'wb') as file:
    pickle.dump(merged_dfs, file)

df_primary = merged_dfs['df_primary_schools']