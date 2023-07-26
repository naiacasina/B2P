import geopandas as gpd
import pickle
import os
import networkx as nx
import pandas as pd
import ast
from scipy.spatial.distance import euclidean

folder = "Uganda"
country = "uganda"

os.chdir(f"/Users/naiacasina/Documents/SEM2/B2P/Data/{folder}/")

nearest_f_health = pd.read_csv(f'Saved data/health_facilities_footpaths.csv')
nearest_f_religious = pd.read_csv(f'Saved data/religious_facilities_footpaths.csv')
nearest_f_primary = pd.read_csv(f'Saved data/primary_schools_footpaths.csv')
nearest_f_secondary = pd.read_csv(f'Saved data/secondary_school_footpaths.csv')
# nearest_f_highpop = pd.read_csv('Saved data/highpop_centers_footpaths.csv')

bridges_df = pd.read_csv(f'Saved data/nearest_bridge_coords.csv')
# Define the pattern to match tuples with numbers
pattern = r'\(.*\)'
# Filter out rows that match the pattern
bridges_df = bridges_df[bridges_df['bridge_point'].str.contains(pattern)]

# Load the graph object
with open('Saved data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

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

# Set a large number for cases where there is no path
large_number = 999999

facility_types = {
    'health': nearest_f_health,
    'religious': nearest_f_religious,
    'primary': nearest_f_primary,
    'secondary': nearest_f_secondary
}

# Iterate over each row in bridges_df
for index, row in bridges_df.iterrows():
    start_pixel = ast.literal_eval(row['footpath_coord_2'])
    print(index)

    for facility_type, facility_df in facility_types.items():
        # Compute Euclidean distances and find the 10 closest facilities
        closest_facilities = facility_df.copy()
        closest_facilities['distance'] = closest_facilities['footpath_coord_1'].apply(
            lambda coord: euclidean(ast.literal_eval(coord), tuple(start_pixel)))
        closest_facilities = closest_facilities.nsmallest(10, 'distance')
        print(facility_type)

        # Initialize variables for shortest path computation
        shortest_dist = float('inf')
        shortest_index = None

        # Iterate over each row in closest_facilities
        for _, facility_row in closest_facilities.iterrows():
            end_pixel = ast.literal_eval(facility_row['footpath_coord_1'])
            # Check if facility node is in the graph
            if end_pixel not in G:
                print(end_pixel)
                end_pixel = compute_nearest_node(G, end_pixel[0], end_pixel[1])
                print('Not in G')
                print(end_pixel)

            try:
                # Compute the shortest path using NetworkX
                dist = nx.shortest_path_length(G, source=start_pixel, target=end_pixel)
            except nx.NetworkXNoPath:
                # Assign a large number when there is no path
                dist = large_number
            except nx.NodeNotFound:
                # Skip this facility if either source or target node is not found
                continue

            if dist < shortest_dist:
                shortest_dist = dist
                shortest_index = closest_facilities.index.get_loc(facility_row.name)

        # Store the shortest path result in the bridges_df dataframe
        column_name = f'shortest_2_{facility_type}'
        bridges_df.at[index, column_name] = shortest_dist

bridges_df.to_csv('Saved data/bridges_shortest_paths_Jun8.csv')
bridges_df.to_csv('Saved data/bridges_shortest_paths.csv')



# --------------- Random waterway points -----------
ww_df = pd.read_csv('Saved data/nearest_ww_coords.csv')
large_number = 999999

facility_types = {
    'health': nearest_f_health,
    'religious': nearest_f_religious,
    'primary': nearest_f_primary,
    'secondary': nearest_f_secondary
}

# Iterate over each row in bridges_df
for index, row in ww_df.iterrows():
    start_pixel = row['footpath_coord_2']
    print(index)

    for facility_type, facility_df in facility_types.items():
        # Compute Euclidean distances and find the 10 closest facilities
        closest_facilities = facility_df.copy()
        closest_facilities['distance'] = closest_facilities['footpath_coord_1'].apply(
            lambda coord: euclidean(ast.literal_eval(coord), eval(start_pixel)))
        closest_facilities = closest_facilities.nsmallest(20, 'distance')
        print(facility_type)

        # Initialize variables for shortest path computation
        shortest_dist = float('inf')
        shortest_index = None

        # Iterate over each row in closest_facilities
        for _, facility_row in closest_facilities.iterrows():
            end_pixel = ast.literal_eval(facility_row['footpath_coord_1'])
            # Check if facility node is in the graph
            if end_pixel not in G:
                print(end_pixel)
                end_pixel = compute_nearest_node(G, end_pixel[1], end_pixel[0])
                print('Not in G')
                print(end_pixel)

            try:
                # Compute the shortest path using NetworkX
                dist = nx.shortest_path_length(G, source=eval(start_pixel), target=end_pixel)
            except nx.NetworkXNoPath:
                # Assign a large number when there is no path
                dist = large_number
            except nx.NodeNotFound:
                # Skip this facility if either source or target node is not found
                continue

            if dist < shortest_dist:
                shortest_dist = dist
                shortest_index = closest_facilities.index.get_loc(facility_row.name)

        print(shortest_dist)
        # Store the shortest path result in the ww_df dataframe
        column_name = f'shortest_2_{facility_type}'
        ww_df.at[index, column_name] = shortest_dist

ww_df.to_csv('Saved data/ww_shortest_paths_Jun8.csv')
ww_df.to_csv('Saved data/ww_shortest_paths.csv')