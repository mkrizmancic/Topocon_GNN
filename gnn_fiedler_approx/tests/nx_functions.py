import contextlib
import errno
import math
import os
import random
import signal
import time

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_TIMEOUT_MESSAGE = os.strerror(errno.ETIME)


class timeout(contextlib.ContextDecorator):
    """
    Easily put time restrictions on things

    Usage as a context manager:
    ```
    with timeout(10):
        something_that_should_not_exceed_ten_seconds()
    ```
    Usage as a decorator:
    ```
    @timeout(10)
    def something_that_should_not_exceed_ten_seconds():
        do_stuff_with_a_timeout()
    ```
    Handle timeouts:
    ```
    try:
    with timeout(10):
        something_that_should_not_exceed_ten_seconds()
    except TimeoutError:
        log('Got a timeout, couldn't finish')
    ```
    Suppress TimeoutError and just die after expiration:
    ```
    with timeout(10, suppress_timeout_errors=True):
        something_that_should_not_exceed_ten_seconds()
    print('Maybe exceeded 10 seconds, but finished either way')
    ```
    """
    def __init__(self, seconds, *, timeout_message=DEFAULT_TIMEOUT_MESSAGE, suppress_timeout_errors=False):
        self.seconds = int(seconds)
        self.timeout_message = timeout_message
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.suppress and exc_type is TimeoutError:
            return True

# Define the list of functions to be tested
functions = {
    'algebraic_connectivity': lambda x: nx.laplacian_spectrum(x)[1],
    'degree': lambda G: G.degree,
    'average_neighbor_degree': nx.average_neighbor_degree,
    'degree_centrality': nx.degree_centrality,
    'eigenvector_centrality': nx.eigenvector_centrality,
    'eigenvector_centrality_numpy': nx.eigenvector_centrality_numpy,
    'katz_centrality': lambda G: nx.katz_centrality(G, 1 / max(nx.adjacency_spectrum(G)) / 2),
    'katz_centrality_numpy': lambda G: nx.katz_centrality_numpy(G, 1 / max(nx.adjacency_spectrum(G) / 2)),
    'closeness_centrality': nx.closeness_centrality,
    'current_flow_closeness_centrality': nx.current_flow_closeness_centrality,
    'betweenness_centrality': nx.betweenness_centrality,
    'betweenness_centrality (approx.)': lambda G: nx.betweenness_centrality(G, k=5),
    'current_flow_betweenness_centrality': nx.current_flow_betweenness_centrality,
    'current_flow_betweenness_centrality (approx.)': nx.approximate_current_flow_betweenness_centrality,
    # 'communicability_betweenness_centrality': nx.communicability_betweenness_centrality,  # Too slow.
    'subgraph_centrality': nx.subgraph_centrality,
    'harmonic_centrality': nx.harmonic_centrality,
    'percolation_centrality': nx.percolation_centrality,
    'second_order_centrality': nx.second_order_centrality,
    'laplacian_centrality': nx.laplacian_centrality,
    'triangles': nx.triangles,
    'clustering': nx.clustering,
    'square_clustering': nx.square_clustering,
    'core_number': nx.core_number,
    'eccentricity': nx.eccentricity,
    'constraint': nx.constraint,
    'effective_size': nx.effective_size,
    # 'closeness_vitality': nx.closeness_vitality,  # Second level too slow.
    'degree_assortativity_coefficient': nx.degree_assortativity_coefficient,
    'degree_pearson_correlation_coefficient': nx.degree_pearson_correlation_coefficient,
    'is_at_free': nx.is_at_free,
    'has_bridges': nx.has_bridges,
    'is_chordal': nx.is_chordal,
    'transitivity': nx.transitivity,
    'average_clustering': nx.average_clustering,
    # 'average_clustering (approx.)': nx.algorithms.approximation.average_clustering,  # Slower than the exact version.
    # 'average_node_connectivity': nx.average_node_connectivity,  # Second level too slow.
    'node_connectivity': nx.node_connectivity,
    'node_connectivity (approx.)': nx.algorithms.approximation.node_connectivity,
    'edge_connectivity': nx.edge_connectivity,
    'girth': nx.girth,
    'diameter': nx.diameter,
    'diameter (approx.)': nx.algorithms.approximation.diameter,
    'effective_graph_resistance': nx.effective_graph_resistance,
    'kemeny_constant': nx.kemeny_constant,
    'radius': nx.radius,
    'is_distance_regular': nx.is_distance_regular,
    'is_strongly_regular': nx.is_strongly_regular,
    'local_efficiency': nx.local_efficiency,
    'global_efficiency': nx.global_efficiency,
    'is_planar': nx.is_planar,
    'is_regular': nx.is_regular,
    # 'sigma': nx.sigma,  # Too slow.
    # 'omega': nx.omega,  # Not well implemented, often fails, and slow.
    's_metric': nx.s_metric,
    'is_tree': nx.is_tree,
    'wiener_index': nx.wiener_index,
    'schultz_index': nx.schultz_index,
    'gutman_index': nx.gutman_index,
}

# Define the graph sizes
graph_sizes = [5, 10, 20, 50, 100]

# Store the results
results = []

# Test each function on each graph size
for size in tqdm(graph_sizes):
    num_graphs = 20  # 5 if size < 50 else 5
    for i in range(num_graphs):
        # Generate a random graph
        G = nx.connected_watts_strogatz_graph(size, random.randint(2, int(math.sqrt(size))), random.random())
        for name, func in tqdm(functions.items(), leave=False):
            try:
                start_time = time.perf_counter()
                with timeout(10):
                    func(G)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
            except TimeoutError:
                tqdm.write(f'Timeout in {name}')
                execution_time = 10
            except Exception as e:
                tqdm.write(f'Error in {name}: {e}')
                execution_time = np.nan
            results.append({
                'Function': name,
                'Graph Size': size,
                'Execution Time': execution_time
            })


# Convert results to DataFrame
df = pd.DataFrame(results)

# Calculate average execution time for each function and graph size
average_execution_time = df.groupby(['Function', 'Graph Size'])['Execution Time'].mean().unstack()

# Calculate total execution duration for each function
total_execution_duration = df.groupby('Function')['Execution Time'].sum().reset_index()
total_execution_duration.columns = ['Function', 'Total']

# Calculate the total execution duration for all functions
total_duration_all_functions = total_execution_duration['Total'].sum()

# Calculate relative duration for the total duration for each function
total_execution_duration['Total_Relative'] = total_execution_duration['Total'] / total_duration_all_functions * 100

# Calculate relative duration with respect to the total duration of all functions for each graph size
relative_duration = average_execution_time.copy()
for size in graph_sizes:
    relative_duration[size] = relative_duration[size] / average_execution_time[size].sum() * 100

# Add relative difference compared to the first row for each function
first_row_total_duration = total_execution_duration.iloc[0]['Total']
total_execution_duration['Baseline'] = (total_execution_duration['Total'] - first_row_total_duration) / first_row_total_duration * 100

# Merge the relative total duration into the relative duration DataFrame
final_results = average_execution_time.reset_index().merge(total_execution_duration, on='Function')
final_results = final_results.merge(relative_duration.reset_index(), on='Function', suffixes=('', '_Relative'))

# Sort final_results based on the order of keys in functions
sorted_functions = list(functions.keys())
final_results = final_results.set_index('Function').loc[sorted_functions].reset_index()

# Round all numerical columns to 5 decimal places
final_results = final_results.round(5)

# Print the results
print(final_results)

# Optional: Save the results to a CSV file
final_results.to_csv('results/nx_function_performance.csv', index=False)
