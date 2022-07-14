# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup PySpark

# %%
# %load_ext sparkmagic.magics

# %%
import os
from IPython import get_ipython
# %load_ext autoreload
# %autoreload 2

username = os.environ['RENKU_USERNAME']
server = "http://iccluster029.iccluster.epfl.ch:8998"

get_ipython().run_cell_magic(
    'spark',
    line='config', 
    cell="""{{ "name": "{0}-steph-aces", "executorMemory": "4G", "executorCores": 4, "numExecutors": 10, "driverMemory": "4G"}}""".format(username)
)

# %%
get_ipython().run_line_magic(
    "spark", "add -s {0}-steph-aces -l python -u {1} -k".format(username, server)
)

# %% language="spark"
# import pyspark.sql.functions as F
# from pyspark.sql.types import StructType

# %% [markdown]
# ## Part I: Importing the basis graph data
# First, we need to load the basis graph, which is pre-built and saved in orc format as a list of edges on HDFS, containing all the stations and all the possible trips.

# %% language="spark"
#
# # Load the basis graph
# edges = spark.read.orc('/group/aces/graph/all_edges_final.orc')

# %% language="spark"
# edges.count()

# %% [markdown]
# First we need to **prune** the graph of all nodes and edges that we are sure won't satisfy the query.

# %% [markdown]
# ## Part II: Pruning the graph 

# %% [markdown]
# Knowing that we can construct the graph directly from the edges dataframe, and knowing that the edges always have the starting node and the destination node, we can do the filtering on the edges directly. To reconstruct the times of departure and arrival at nodes, the edges encode it in the node id itself `full_stop_id`.
#
# Let's first define a function that extracts the departure and arrival times from these nodes and add them as new columns in the edges dataframe. We also convert them to timestamp.
# Here we use `to_timestamp` to convert times to timestamps that we can work with.

# %% [markdown]
# ### Part II.a: Pruning based on the arrival time
#
# Since we have a fixed required arrival time, we can filter out all the edges whose arrival or departure time is later than this arrival time. We extract these times from the full_stop_id.

# %% language="spark"
#
# @F.udf
# def extract_time_from_id(full_stop_id):
#     """
#     Helper function to extract the time from the stop_id. The time is encoded as the token that comes after 
#     the first underscore.
#     
#     :param full_stop_id: full stop id encoding the time
#     :return time as a string HH-mm-ss
#     """
#     return full_stop_id.split('_')[1]
#
#
# def filter_before_arrival_time(sdf_edges, arrival_time):
#     """
#     Filter the edges having a departure or arrival time later
#     than the required arrival time.
#     
#     :param sdf_edges: spark dataframe containing the edges
#     :param arrival_time: required arrival time
#     :return spark dataframe with the remaining edges
#     """
#     # Convert all times to timestamp to be able to compare, add or subtract
#     time_format = 'HH-mm-ss'
#     edges_filtered = edges.withColumn('time_departure', F.to_timestamp(extract_time_from_id('start_id'), time_format)) \
#                            .withColumn('time_arrival', F.to_timestamp(extract_time_from_id('end_id'), time_format))
#     
#     edges_filtered = edges_filtered.withColumn('desired_arrival_time', F.to_timestamp(F.lit(arrival_time), 'HH:mm:ss'))
#     
#     # Filter out all edges where arrival time is after desired arrival time T
#     filter_1 = edges_filtered.time_arrival > edges_filtered.desired_arrival_time
#
#     # Filter out all edges where departure time is after desired arrival time T
#     filter_2 = edges_filtered.time_departure > edges_filtered.desired_arrival_time
#
#     return edges_filtered.filter(~(filter_1 | filter_2))
#     
#
# def filter_too_early_edges(sdf_edges, arrival_time, max_duration=2):
#     """
#     Filter the edges having a departure time earlier than the
#     arrival time minus max trip duration. This removes edges
#     that are too early w.r.t. the arrival time to be used
#     by a good trip.
#     
#     :param sdf_edges: spark dataframe containing the edges
#     :param arrival_time: required arrival time
#     :param max_duration: maximum duration of a trip in hours
#     :return spark dataframe with the remaining edges
#     """
#     # To add duration: https://sparkbyexamples.com/spark/spark-add-hours-minutes-and-seconds-to-timestamp
#     return edges_filtered.filter(
#         edges_filtered.time_departure > edges_filtered.desired_arrival_time - F.expr('INTERVAL {} HOURS'.format(max_duration))
#     )

# %% [markdown]
# Below, we do an example where we set the arrival time at 10:00.

# %% language="spark"
#
# # Example desired arrival time: 10:00am
# arrival_time = '10:00:00'
# edges_filtered = filter_before_arrival_time(edges, arrival_time)
#
# # Uncomment to see results
# # edges_filtered.show(n=10)

# %% language="spark"
# edges_filtered.count()

# %% [markdown]
# We can see that we have already much less edges than in the original graph.

# %% [markdown]
# ### Part II.b: Pruning based on maximum trip duration assumptions

# %% [markdown]
# Another way to prune our graph is to figure out a **reasonable time to start your journey**. For example if we want to arrive before 9pm, we can't expect you to take the bus at 9am.
# For this we set a time limit at which we start a journey.

# %% language="spark"
#
# # Check other simplifying assumptions about the data (e.g. check for maximal duration of a trip and filter out trips that depart before that duration). 
# # We're trying to mimize the travel time as well!
# max_trip_duration = edges.agg(F.max(edges.duration)).collect()[0]
# print('Maximum duration for trip between 2 stations is: {} minutes'.format(max_trip_duration[0] / 60))

# %% [markdown]
# First we see that the maximum trip time between any 2 stops is around ?? minutes. We also searched for different trips between the extremities of the circle on Google maps and looked at their duration. We then decided to set the time limit to start our journey at the earliest to **2 hours before the expected arrival time** for now. We remove the edges having a departure time that is earlier than the arrival time minus the maximum duration set.

# %% language="spark"
#
# # Filter out edges that are too early to be considered
# edges_filtered = filter_too_early_edges(edges_filtered, arrival_time, max_duration=2)

# %% [markdown]
# Let's check how many edges we have left.

# %% language="spark"
# edges_filtered.count()

# %% [markdown]
# To be able to construct the graph we need to import the graph dataframe from spark into local notebook to be able to run it through networkx. One way to get back the dataframe into the local context is to use the ```-o``` option in the spark context. However, as the dataframe contains multiple hundreds of thousands of rows, it is very slow to send it through the network. We rather write it into a csv file on the HDFS which we will be able to read from the local context.

# %% language="spark"
#
# def save_csv_to_hdfs(sdf_edges, file):
#     """
#     Save the spark dataframe containing the edges as a csv file on the HDFS
#     in the given file.
#     
#     :param sdf_edges: spark dataframe containing the edges
#     :param file: file name
#     """
#     sdf_edges.write.csv('{0}'.format(file), mode='overwrite', header='true')

# %% language="spark"
#
# # Save to HDFS
# save_csv_to_hdfs(edges_filtered, '/group/aces/cache/edges_filtered.csv')

# %% [markdown]
# ## Part III: Constructing the graph

# %% [markdown]
# ### Part III.a - Load the pruned graph from HDFS
#
# We first need to read the pruned graph from the csv files on the HDFS.

# %%
from hdfs3 import HDFileSystem
import pandas as pd
import pickle


def read_file_from_hdfs(file, compressed=False):
    """
    Read a csv file from the HDFS into
    a "local" pandas dataframe
    
    :param file: file to read on HDFS
    :param compressed: whether the file is compressed
    :return pandas dataframe
    """
    # HDFS explorer
    hdfs = HDFileSystem()

    # File is split into multiple small files
    results = []
    
    # Iterate over all the small files
    for path in hdfs.ls(file):
        if not 'SUCCESS' in path:
            
            # Open the small file
            with hdfs.open(path) as f:
                if compressed:
                    results.append(pd.read_csv(f, compression='bz2'))
                else:
                    results.append(pd.read_csv(f, dtype={'line_number': str}))

    # Concatenate the content of all the small files
    # into a single pandas dataframe
    return pd.concat(results)


def clean_folder_hdfs(folder_path):
    """
    Delete all the files in the given folder
    
    :param folder_path: path to the folder to clean
    """
    hdfs = HDFileSystem()
    
    hdfs.rm(folder_path)
    

def put_on_hdfs(list_object, file):
    
    hdfs = HDFileSystem()
    
    with hdfs.open(file, 'wb') as f:
        pd.Series(list_object).to_csv(f)


# %%
# Read the graph as an edge list in a pandas dataframe
df_edges = read_file_from_hdfs('/group/aces/cache/edges_filtered.csv')

# %%
## Uncomment to see results
# df_edges.head()

# %% [markdown]
# ### Part III.b - Construct the graph using Networkx
#
# Now, we build a networkx graph based on the pruned edges list.

# %%
# install networkx package
# !pip install networkx

# %%
import networkx as nx

def build_graph(df_edges):
    """
    Build a networkx graph based on the
    edges list in the given pandas dataframe
    
    :param df_edges: pandas dataframe containing the edges list
    :return networkx directed graph
    """
    return nx.from_pandas_edgelist(df_edges, source='start_id', target='end_id', edge_attr='edge_weight_wt_mean_delay', create_using=nx.DiGraph())


# %%
# %%time
# Uncomment this line to see the time it takes to build the graph
# Build the graph with the pruned edges list
# G = build_graph(df_edges)

# %% [markdown]
# ## Part IV: Interpret the query function

# %% [markdown]
# ### Build the pruning and networkx graph building pipeline
# We now define the pipeline that reads the basis graph, prune it as explained before and finally build the networkx graph that will then be used to compute the shortest paths.

# %%
def run_spark(cmd, line=''):
    """
    Run code inside the spark context
    
    :param cmd: code to run given as a string
    """
    get_ipython().run_cell_magic(
    magic_name="spark", line=line,
    cell=cmd
    )


def pruning_building_pipeline(basis_graph_file, arrival_time, max_duration):
    """
    Run the enire pruning and networkx graph building pipeline
    
    :param basis_graph_file: orc file containing the basis graph edges list
    :param arrival_time: required arrival time
    :param max_duration: maximum trip duration
    :return networkx pruned directed graph, pandas dataframe containing the edges list
    """
    # Read the basis graph
    # Filter the edges based on arrival time
    # Filter the edges based on maximum trip duration
    # Save the pruned graph on HDFS
    pruned_graph_file = '/group/aces/cache/edges_filtered.csv'
    
    run_spark(
        """
        edges = spark.read.orc('/group/aces/{0}')
        edges_filtered = filter_before_arrival_time(edges, '{1}')
        edges_filtered = filter_too_early_edges(edges_filtered, '{1}', max_duration={2})
        #print(edges_filtered.schema)
        save_csv_to_hdfs(edges_filtered, '{3}')
        callback_df = spark.createDataFrame([], StructType([]))
        """.format(basis_graph_file, arrival_time, max_duration, pruned_graph_file),
        line='-o callback_df'
    )
    
    # Read the pruned graph in the local context
    df_edges = read_file_from_hdfs(pruned_graph_file)
    
    return build_graph(df_edges), df_edges


# %% [markdown]
# Now we define the **query function** that will be called when searching for a route. It takes 4 arguments:
# - The departure station A
# - The destination station B
# - The max arrival time T at or before which we'd like to arrive to our destination
# - The probability or confidence Q that the user wishes to have to arrive at destination B before time T
# - The maximum number of paths we want to output

# %% tags=[]
from itertools import islice
from datetime import datetime
from confidence import path_confidence
import numpy as np


def fetch_nodes_infos(file, nodes_all_info_file):
    
    # Read the csv list of the nodes sequences for
    # each shortest paths.
    # Load the spark dataframe containing all the info
    # Only keep info of nodes present in paths
    run_spark(
        """
        from pyspark.sql.functions import col
        
        paths_nodes = [row[1] for row in spark.read.csv('{0}', header='true').collect()]

        all_nodes_infos = spark.read.orc('{1}')
        df_nodes_info = all_nodes_infos.filter(col('full_stop_id').isin(paths_nodes))
        """.format(file, nodes_all_info_file),
        line='-o df_nodes_info'
    )
    return df_nodes_info


def add_source_sink(G, start_id, end_id, arrival_time):
    """
    Add a source node connecting all the possible start nodes
    and a sink node connecting all the possible end nodes. Both
    have weight 0.
    
    :param G: pruned directed graph
    :param start_id: start stop id
    :param end_id: end stop id
    :param arrival_time: required arrival time at destination
    :return networkx directed graph, source node id, sink node id
    """
    src = 'source_node'
    sink = 'sink_node'
    # Add source node
    G.add_node(src)
    
    # Add sink node
    G.add_node(sink)
    
    arrival_time_dt = datetime.strptime(arrival_time, '%H:%M:%S')
    
    # Go over all nodes and add if it is the start or end stop_id
    for n in G.nodes():
        if (n != 'source_node') and (n != 'sink_node'):
            # Get the stop_id of the node
            full_id_split = n.split('_')
            
            # Get only the stop_id without platform information
            stop_id = full_id_split[0].split(':')[0]
            time = full_id_split[1]

            if stop_id == start_id:
                # O weight from source node to start nodes
                G.add_edge(src, n, edge_weight_wt_mean_delay=0)

            elif stop_id == end_id:
                # Waiting time from arrival with transport until arrival_time
                time_dt = datetime.strptime(time, '%H-%M-%S')
                delta = (arrival_time_dt - time_dt).total_seconds()
                
                G.add_edge(n, sink, edge_weight_wt_mean_delay=delta)
            
    return G, src, sink

def paths_confidence(paths_nodes, df_edges, certainty, G):
    """
    Compute the confidence for all the paths characterized by lists
    of nodes id. Only keep the paths having confidence above 'certainty'.
    
    :param paths_nodes: list of list of nodes id
    :param df_edges: pandas dataframe of edges list
    :param certainty: required confidence for the path
    :return list of tuples (list(edges), path confidence)
    """
    result = []
    
    # Compute the confidence for each path
    for path in paths_nodes:
        # Sequence of edges (tuples (depart stop_id, target stop_id)) in path
        path_edges = list(zip(path[1:], path[:-1]))
        
        free_time = G.get_edge_data(path_edges[0][0], path_edges[0][1])['edge_weight_wt_mean_delay']
        
        # Remove the edges from source and to sink
        path_edges = path_edges[1:-1]
        
        # Compute the confidence of the path
        path_conf = path_confidence(df_edges.loc[path_edges], free_time)
        
        if path_conf >= certainty:
            result.append((path_edges[::-1], path_conf, path[1:-1][::-1]))
            
    return result


def search_graph(start_id, end_id, arrival_time, certainty, k=10):
    """
    Get the k-shortest path from start_id station to the end_id station before the requested arrival hour.
    :param start_id: the id of the departure station
    :param end_id: the id of the destination station
    :param arrival_time: the hour at which we need to be in the destination station.
    :param certainty: the certainty level at which the trip will be finished.
    :param k: the number of different shortest paths the function needs to return
    :return: a dataframe with trip information            
        - df: (detailed trip) contains one line for each stop 
            - num_shortest_path: (0 is the fastest, to k-1)
            - subtrip_sequence: the position of this (sub)trip in the shortest path 
            - start_name: the name of the start station for this (sub)trip
            - end_name: the name of the destination station for this (sub)trip
            - lat: latitude of the stop
            - lon: longitude of the stop
            - start_hour: the hour at which the transport starts from the start station.
            - end_hour: the hour at which the transport arrives in the destination station.
            - transport: the type of transport used
            - line_number: the line number of the used transport
            
        E.g. if we have a trip A -> B -> C using B13, C -> D -> E using B15, 4 lines in df, i.e. A->B using B13, B->C using B13, C->D using B15, D->E using B15
    """
    arrival_time = arrival_time[:6] + '59'
    # Get the pruned networkx graph for this query
    G, df_edges = pruning_building_pipeline('graph/all_edges_final.orc', arrival_time, max_duration=2)
    df_edges = df_edges.set_index(['start_id', 'end_id'])
    
    # Add the source and sink nodes for the shortest paths
    G, src, sink = add_source_sink(G, start_id, end_id, arrival_time)
    
    # Find the k shortest paths, results is a list of k lists containing a sequence of nodes
    # representing each path
    try:
        # Search paths starting from destination
        
        paths_nodes = list(islice(nx.shortest_simple_paths(G.reverse(), sink, src, weight='edge_weight_wt_mean_delay'), k))
        #paths_nodes = [nx.shortest_path(G.reverse(), sink, src, weight='edge_weight_wt_mean_delay')]
        
        # Compute the confidence for each path
        paths_confidences = paths_confidence(paths_nodes, df_edges, certainty, G)
    
    except nx.NetworkXNoPath:
        # Did not found any path
        print('No path were found.')
        return pd.DataFrame(), pd.DataFrame(), [], G
    
    if len(paths_confidences) == 0:
        # No path had enough confidence
        print('No path has the required confidence.')
        return pd.DataFrame(), pd.DataFrame(), [], G
    
    # Compute set of unique nodes visited
    nodes_set = list(set([n for path in paths_nodes for n in path]))
    
    # Write them to hdfs
    nodes_set_file = '/group/aces/graph/nodes_set.csv'
    put_on_hdfs(nodes_set, nodes_set_file)
    
    # Fetch the infos for the nodes in the paths
    df_nodes_info = fetch_nodes_infos(nodes_set_file, '/group/aces/graph/nodes_all_info_final.orc').set_index('full_stop_id')

    result_edges = []
    result_nodes = []
    result_conf = []
    idx_sorted = np.argsort([x[1] for x in paths_confidences])[::-1]
    already_seen = set()
    
    idx = 0
    for i in idx_sorted:
        path_edges, conf, path_nodes = paths_confidences[i]
        
        # If we have already seen this path
        if conf not in already_seen:
            already_seen.add(conf)
            
            # Shortest paths were computed from sink
            df_path = df_edges.loc[path_edges]

            # Add index of this path
            df_path['num_shortest_path'] = idx
            idx += 1
            df_path['subtrip_sequence'] = list(range(df_path.shape[0]))

            df_path_nodes = df_nodes_info.loc[path_nodes]
            df_path_nodes['num_shortest_path'] = i
            df_path_nodes['subtrip_sequence'] = list(range(df_path_nodes.shape[0]))

            result_edges.append(df_path)
            result_nodes.append(df_path_nodes)
            result_conf.append(conf)
        
    # Add the corresponding nodes info for the map
    df_nodes_info_add = df_nodes_info[['stop_lat', 'stop_lon', 'time','stop_name']]
    result_edges = pd.concat(result_edges)
    result_edges = result_edges.join(df_nodes_info_add, on='start_id', how='left', rsuffix='_start')
    result_edges = result_edges.join(df_nodes_info_add, on='end_id', how='left', rsuffix='_end')
    result_edges = result_edges.rename(columns={'stop_lat': 'stop_lat_start', 'stop_lon': 'stop_lon_start',
                                   'time': 'time_start', 'stop_name': 'stop_name_start'})
    
    result_nodes = pd.concat(result_nodes)
    
    return result_edges, result_nodes, result_conf, G


# %%
# %%time

# Uncomment this line to see the results of the graph route search
# df_edges, df_nodes, confidences, G = search_graph('8503052', '8591202', '13:15:00', 0.0, k=10)

# %% [markdown]
# # Part V: Graphical Interface

# %%
from IPython.display import clear_output
from ipywidgets import Box, widgets, Layout, VBox, HBox
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random

# %% [markdown]
# ## Load the Zürich stops data for display on the interface 

# %% [markdown]
# We've extracted all the Zürich stops (in a separate local csv file since it was small enough to store locally), to link their ids to their names when displaying them on the interface.

# %%
zurich_stops = pd.read_csv('../data/stops_infos.csv',  dtype = {'stop_id': str, 'stop_name': str, 'c': 'Float64', 'stop_lon': 'Float64'})
print(zurich_stops.shape[0])
zurich_stops.head()


# %% tags=[]
# helper funcitons for mappings between station name, station id and (lat,lon) of the stations
def station_name_from_id(station_id):
    return zurich_stops[zurich_stops['stop_id'] == station_id].iloc[0]['stop_name']

def station_id_from_name(station_name):
    return zurich_stops[zurich_stops['stop_name'] == station_name].iloc[0]['stop_id']

def latlon_from_name(station_name):
    lat = zurich_stops[zurich_stops['stop_name'] == station_name].iloc[0]['stop_lat']
    lon = zurich_stops[zurich_stops['stop_name'] == station_name].iloc[0]['stop_lon']
    return lat, lon


# %%
def find_paths(from_station, to_station, arrival_hour, certainty, k=10):
    '''
    Prepares the input for search_graph() function
    
    :param from_station: name of the start station (string)
    :param to_station: name of the end funciton (string)
    :param arrival_hour: datetime in '%H:%M' format
    :param certainty: the minimum certainty of the path (float from 0 to 100)
    :param k: the number of the routes to find (default 10)
    :return: the dataframe containing k shortest paths from from_station to to_station (see search_graph() funciton)
    '''
    # find station ids 
    # there is hopefully no duplicates in the station names
    start_id = station_id_from_name(from_station)
    end_id = station_id_from_name(to_station)
    
    arrival_time = datetime.strftime(arrival_hour, '%H:%M:%S')
    
    # certainty % -> [0, 1]
    certainty /= 100
    
    return search_graph(start_id, end_id, arrival_time, certainty, k)


# %% [markdown]
# ## Widgets

# %% tags=[]
def routesWidget(shortest_routes, certainties=[]):
    '''
    Returns a Tab widget containing routes 
    :param routes: pandas dataframe of the format returned by search_graph() method
    :return tabs: the ipywidgets.widgets.Tab 
        - each tab corresponds to n'th shortest path and contains an accordion
        - each tab of the accordion corresponds to the subtrip 
        - the body of the accordion contains the intermediate stations in the subtrip
    
    '''
    routes = shortest_routes.copy()
    routes['line_number'] = routes['line_number'].astype('int32', copy=False)
    routes = routes[~((routes['transport'] == 'feet') & (routes['line_number'] > 0))]
    
    # Assign different line number to each walking edge to be able to distinguish them when grouping transports
    walking_edges = routes[routes.transport == 'feet']
    line_numbers_distinct = [-i for i in range(len(walking_edges))]
    routes.loc[routes.transport == 'feet', 'line_number'] = line_numbers_distinct
    
    
    routes = routes.sort_values(by=['num_shortest_path', 'subtrip_sequence'])   
    routes['start_hour'] = (pd.to_datetime(routes['start_hour'], format='%H:%M')).dt.strftime('%H:%M') 
    routes['end_hour'] = pd.to_datetime(routes['end_hour'], format='%H:%M').dt.strftime('%H:%M') 
    
    # routes tab
    tabs = widgets.Tab(layout=Layout(margin='20px'))
    tabs_children = []
    route_nbs = routes['num_shortest_path'].unique()
    
    for i, rte_nb in enumerate(route_nbs):
        tabs.set_title(i, f'Route {i}')
        
        # df containing route #rte_nb only
        route = routes[routes['num_shortest_path'] == rte_nb]
        
        # accordion containing the intermediate stops of the subtrip
        route_acc = widgets.Accordion()
        route_acc_children = []
        route_acc_titles = []
        
        # unique (transport, line_number) tuples
        transport_line = route[['transport', 'line_number']].drop_duplicates()
        

        # subtrip details
        for (transport, line_number) in (transport_line.values):
            subroute = route[(route.transport == transport) & (route.line_number == line_number)]
            html = ''
            for idx, row in subroute.iterrows():
                start_hour = row['start_hour']
                end_hour = row['end_hour']
                start_name = row['start_name']
                end_name = row['end_name']
                html += f'<div>{start_hour} {start_name} -> {end_hour} {end_name}</div>'
            route_acc_children.append(widgets.HTML(html))
            
        route_acc.children = route_acc_children
        
        # data for the accordion titles (first and last line per subtrip)
        subroute_groupped = route.groupby(['transport', 'line_number'], sort=False)
        subroute_start_names = subroute_groupped.head(1)['start_name'].values
        subroute_end_names = subroute_groupped.tail(1)['end_name'].values
        subroute_start_hours = subroute_groupped.head(1)['start_hour'].values
        subroute_end_hours = subroute_groupped.tail(1)['end_hour'].values
        subroute_transports = subroute_groupped.head(1)['transport'].values
        subroute_lines = subroute_groupped.head(1)['line_number'].values
        
        nb_subroutes = subroute_start_names.shape[0]
        
        # set the accordion titles
        for j in range(nb_subroutes):
            if subroute_transports[j] == 'feet':
                line_number = ''
            else :
                line_number = ' ' + str(subroute_lines[j])
            route_acc.set_title(j, f'{subroute_start_hours[j]} {subroute_start_names[j]} ---> \
                                {subroute_end_hours[j]} {subroute_end_names[j]} \
                                [{subroute_transports[j]}{line_number}]')
    
        
        tab_content = route_acc
        if certainties:
            cert = certainties[i]
            c = widgets.HTML(f'<div><b>Certainty: {cert*100:.2f}%</b><div>')
            tab_content = VBox([c, route_acc])
        tabs_children.append(tab_content)
        
    tabs.children = tabs_children
    return tabs


# %%
def dateTimePicker():
    '''
    Returns a widget for time (hour, minute) selection
    
    :timepicker
        - timepicker.children[0] is an ipywidgets.widgets.Dropdown corresponding to hours
        - timepicker.children[1] is an ipywidgets.widgets.Dropdown corresponding to minutes
    '''
    
    style = {'description_width': '100px'}
    layout = {'width': '400px'}
    
    # define the time range 
    # hours: from 6h to 20h, minutes: 60 minutes
    hrs = [str(i).rjust(2, '0') for i in range(6, 21)]
    mins = [str(i).rjust(2, '0') for i in range(60)]
    
    hours = widgets.Dropdown(
        description = 'Time:',
        options = hrs, 
        layout = Layout(width='160px')
    )
    hours.style = style

    
    minutes = widgets.Dropdown(
        description = ':',
        options = mins, 
        layout = Layout(width='70px')
    )
    minutes.style = {'description_width': '10px'}
    
    timePicker = HBox([hours, minutes])
    
    return timePicker


# %% [markdown]
# ## Interface
# __Note:__ The map might take a few minutes to generate

# %%
### Data ###
stops = zurich_stops['stop_name'].sort_values().unique()

### Map ###
def draw_map(route_nodes):
    '''
    Constructs the map containing the shortest routes
    
    :param route_nodes: pandas dataframe containing the shortest path (returned by search_graph() function)
    :returns fig: the plotly express map, line per path
    '''
    center_lat = route_nodes['stop_lat'].mean()
    center_lon = route_nodes['stop_lon'].mean()

    fig = px.scatter_mapbox(route_nodes, lat="stop_lat", lon="stop_lon", 
                            zoom=3, height=300, 
                            hover_name='stop_name')
    
    fig.update_layout(mapbox_style='open-street-map', 
                  mapbox_zoom = 12, 
                  margin={"r":0,"t":20,"l":0,"b":0})
    
    fig.add_traces(px.line_mapbox(route_nodes, lat="stop_lat", lon="stop_lon", 
                                  color="color", 
                                  hover_data=['stop_name', 'time', 'route_desc'], 
                                  center={'lat': center_lat, 'lon': center_lon}, zoom=15).data)
    fig.update_traces(mode="markers+lines")
#     first_path_index = str(route_nodes.iloc[0]['num_shortest_path'])
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly") if trace.name != '0' else ())
    
    return fig


fig = None

### Widgets ###
fromWidget = widgets.Combobox(
    placeholder='From station',
    options=list(stops),
    description='From:',
    ensure_option=True,
    disabled=False
)

toWidget = widgets.Combobox(
    placeholder ='To station',
    options=list(stops),
    description='To:',
    ensure_option=True,
    disabled=False
)

toDateTime = dateTimePicker()

findRoutesButton = widgets.Button(
    description='Find Routes',
    disabled=False,
    tooltip='Find routes')

certaintyText = widgets.BoundedFloatText(
    value=90,
    min=0,
    max=100.0,
    step=0.1,
    description='Certainty:',
    disabled=False
)

out = widgets.Output()
out1 = widgets.Output()
tabs = None

### Layout & style ###
style = {'description_width': '100px'}
layout = {'width': '400px'}

fromWidget.layout = layout
fromWidget.style = style
toWidget.layout = layout
toWidget.style = style
toDateTime.layout = layout
toDateTime.style = style
certaintyText.layout = layout
certaintyText.style = style

### Events ###
def findButtonClicked(b):
    print('button clicked')
    
    from_station = fromWidget.value
    to_station = toWidget.value
    
    from_options = fromWidget.options
    to_options = toWidget.options
    
    with out1:
        # correctness checks
        if len(from_station) == 0:
            print('From station unspecified')
            return

        if len(to_station) == 0:
            print('To station unspecified')
            return

        if from_station == to_station:
            print(f'From and To stations are the same: {from_station}')
            return

        if from_station not in from_options:
            print(f'{from_station} unknown')
            return

        if to_station not in to_options:
            print(f'{to_station} unknown')
            return 

    # convert time to datetime
    to_hour = toDateTime.children[0].value
    to_min = toDateTime.children[1].value
    to_datetime = to_hour + ':' + to_min
    to_datetime = datetime.strptime(to_datetime, '%H:%M')
    certainty = certaintyText.value
    
    out.clear_output()
    out1.clear_output()
    with out1:
        print(f'...searching for routes from [{from_station}] to [{to_station}]...')
        # find the best routes
    df_edges, df_nodes, confidences, G = find_paths(from_station, to_station, to_datetime, certainty, k=10)

    if df_edges.shape[0] == 0:
        with out1:
            print('...no routes found.')
        return
    with out1:
        print('...routes found.')
    routes = df_edges.copy()

    # rename columns to match the naming convention for widgets
    routes = routes.rename(columns={'time_start': 'start_hour', 
                                    'time_end': 'end_hour', 
                                    'stop_lat_start': 'start_lat', 
                                    'stop_lat_end': 'end_lat', 
                                    'stop_lon_start': 'start_lon', 
                                    'stop_lon_end': 'end_lon', 
                                    'stop_name_start':'start_name', 
                                    'stop_name_end': 'end_name'})

    # create the 'color' column for the map
    route_nodes = df_nodes.copy()
    dist_path_nb = route_nodes['num_shortest_path'].unique()
    route_nodes['color'] = 0
    for i, rte_nb in enumerate(dist_path_nb):
        route_nodes.loc[route_nodes.num_shortest_path == rte_nb, 'color'] = i

    print('...building widgets...')
    # build the tabs widget and update the output
    tabs = routesWidget(routes, confidences)
    print('...widgets built.')
    out.clear_output()
    clear_output()
    with out:
        display(tabs)
    fig = draw_map(route_nodes)
    out1.clear_output()
    with out1:
        fig.show()

findRoutesButton.on_click(findButtonClicked)

interface = VBox([fromWidget, toWidget, toDateTime, certaintyText, findRoutesButton])


out.clear_output()
out1.clear_output()


display(interface)
display(VBox([out, out1]))


# %%
