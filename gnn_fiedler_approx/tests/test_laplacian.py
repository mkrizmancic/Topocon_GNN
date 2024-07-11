import time
import networkx as nx
import tqdm
import random
import math
import numpy as np


def custom_l2(G):
    A = nx.to_numpy_array(G)
    D = np.diag(A.sum(1))
    L = D - A
    lambdas = np.linalg.eigvalsh(L)
    l2 = np.sort(lambdas)[1]
    return l2

functions = {
    "algebraic_connectivity": nx.algebraic_connectivity,
    "laplacian_spectrum": lambda x: nx.laplacian_spectrum(x)[1],
    "laplacian_matrix": lambda x: np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(x).todense()))[1],
    "custom": custom_l2
}

total_duration = dict.fromkeys(functions.keys(), 0.0)

num_graphs = 1000
num_tries = 10

for i in tqdm.trange(num_graphs):
    size = random.randint(5, 100)
    # Generate a random graph
    G = nx.connected_watts_strogatz_graph(size, random.randint(2, int(math.sqrt(size))), random.random())

    for name, func in functions.items():
        for j in range(num_tries):
            start = time.perf_counter()
            l2 = func(G)
            end = time.perf_counter()
            total_duration[name] += (end - start)

for name, value in total_duration.items():
    print(f"{name}: {value / (num_graphs * num_tries)}")
