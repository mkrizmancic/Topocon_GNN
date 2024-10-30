"""
MIT License

Copyright (c) 2024 Giannis Nikolentzos
Copyright (c) 2024 Marko Križmančić

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse

import numpy as np
from grakel import Graph
from grakel.kernels import ShortestPath, VertexHistogram, WeisfeilerLehman
from scipy.linalg import svd
import torch
import csv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx



def pyramid_match_kernel(Us, d=20, L=4):
    N = len(Us)

    Hs = {}
    for i in range(N):
        n = Us[i].shape[0]
        Hs[i] = []
        for j in range(L):
            l = 2**j
            D = np.zeros((d, l))
            T = np.floor(Us[i] * l)
            T[np.where(T == l)] = l - 1
            for p in range(Us[i].shape[0]):
                if p >= n:
                    continue
                for q in range(Us[i].shape[1]):
                    D[q, int(T[p, q])] = D[q, int(T[p, q])] + 1

            Hs[i].append(D)

    K = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            k = 0
            intersec = np.zeros(L)
            for p in range(L):
                intersec[p] = np.sum(np.minimum(Hs[i][p], Hs[j][p]))

            k = k + intersec[L - 1]
            for p in range(L - 1):
                k = k + (1.0 / (2 ** (L - p - 1))) * (intersec[p] - intersec[p + 1])

            K[i, j] = k
            K[j, i] = K[i, j]

    return K


class SEGK(torch.nn.Module):
    def __init__(self, radius: int = 2, dim: int = 10, kernel: str = "shortest_path"):
        super().__init__()

        self.radius = radius
        self.dim = dim

        # Set up the graph kernel.
        if kernel == "shortest_path":
            self.gk = [ShortestPath(normalize=True, with_labels=True) for i in range(radius)]
        elif kernel == "weisfeiler_lehman":
            self.gk = [WeisfeilerLehman(n_iter=4, normalize=True, base_graph_kernel=VertexHistogram) for i in range(radius)]
        else:
            raise ValueError("Use a valid kernel!!")

        self.embeddings = None
        self.nodes = None

    def forward(self, data):
        # Convert data from PyG format to NetworkX and extract nodes and edges.
        G = to_networkx(data, to_undirected=True)
        nodes = list(G.nodes)
        self.nodes = nodes
        edgelist = list(G.edges(data=False))
        n = len(nodes)

        # Get the sampled nodes and the remaining nodes for SEGK
        idx = np.random.permutation(n)
        sampled_nodes = [nodes[idx[i]] for i in range(self.dim)]
        remaining_nodes = [nodes[idx[i]] for i in range(self.dim, n)]

        # Extract egonets for the graph
        egonet_edges, egonet_node_labels = self.extract_egonets(edgelist, self.radius)

        # Initialize embeddings tensor
        embeddings = np.zeros((n, self.dim))

        # Kernel computation for sampled nodes
        K = np.zeros((self.dim, self.dim))
        K_prev = np.ones((self.dim, self.dim))
        for i in range(1, self.radius + 1):
            Gs = []
            for node in sampled_nodes:
                node_labels = {
                    v: egonet_node_labels[node][v] for v in egonet_node_labels[node] if egonet_node_labels[node][v] <= i
                }
                edges = []
                for edge in egonet_edges[node]:
                    if edge[0] in node_labels and edge[1] in node_labels:
                        edges.append((edge[0], edge[1]))
                        edges.append((edge[1], edge[0]))
                Gs.append(Graph(edges, node_labels=node_labels))

            K_i = self.gk[i - 1].fit_transform(Gs)
            K_i = np.multiply(K_prev, K_i)
            K += K_i
            K_prev = K_i

        # Perform SVD on the kernel matrix
        U, S, V = svd(K)
        S = np.maximum(S, 1e-12)
        Norm = np.dot(U * 1.0 / np.sqrt(S), V)

        # Perform SVD on the kernel matrix
        embeddings[idx[:self.dim], :] = np.dot(K, Norm.T)

        # Kernel computation for remaining nodes
        K = np.zeros((n - self.dim, self.dim))
        K_prev = np.ones((n - self.dim, self.dim))
        for i in range(1, self.radius + 1):
            Gs = []
            for node in remaining_nodes:
                node_labels = {
                    v: egonet_node_labels[node][v] for v in egonet_node_labels[node] if egonet_node_labels[node][v] <= i
                }
                edges = []
                for edge in egonet_edges[node]:
                    if edge[0] in node_labels and edge[1] in node_labels:
                        edges.append((edge[0], edge[1]))
                        edges.append((edge[1], edge[0]))
                Gs.append(Graph(edges, node_labels=node_labels))

            K_i = self.gk[i - 1].transform(Gs)
            K_i = np.multiply(K_prev, K_i)
            K += K_i
            K_prev = K_i

        # Embedding computation for remaining nodes
        embeddings[idx[self.dim:], :] = np.dot(K, Norm.T)

        # Write embeddings to the data object
        return torch.tensor(embeddings, dtype=torch.float)

    def write_to_file(self, path, delimiter=' '):
        if self.embeddings is None or self.nodes is None:
            raise ValueError("No embeddings to write to file.")

        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter=delimiter)
            for i,node in enumerate(self.nodes):
                lst = [node]
                lst.extend(self.embeddings[i,:].tolist())
                writer.writerow(lst)

    @staticmethod
    def extract_egonets(edgelist, radius, node_labels=None):
        nodes = list()
        neighbors = dict()
        for e in edgelist:
            if e[0] not in neighbors:
                neighbors[e[0]] = [e[1]]
                nodes.append(e[0])
            else:
                neighbors[e[0]].append(e[1])

            if e[1] not in neighbors:
                neighbors[e[1]] = [e[0]]
                nodes.append(e[1])
            else:
                neighbors[e[1]].append(e[0])

        egonet_edges = dict()
        egonet_node_labels = dict()
        for node in nodes:
            egonet_edges[node] = set()
            egonet_node_labels[node] = {node: 0}

        for i in range(1, radius + 1):
            for node in nodes:
                leaves = [v for v in egonet_node_labels[node] if egonet_node_labels[node][v] == (i - 1)]
                for leaf in leaves:
                    for v in neighbors[leaf]:
                        if v not in egonet_node_labels[node]:
                            egonet_node_labels[node][v] = i
                            egonet_edges[node].add((v, leaf))
                            for v2 in neighbors[v]:
                                if v2 in egonet_node_labels[node]:
                                    egonet_edges[node].add((v, v2))

        return egonet_edges, egonet_node_labels


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-edgelist", default="gnn_fiedler_approx/custom_models/datasets/karate.edgelist", help="Path to the edgelist.")
    parser.add_argument("--delimiter", default=" ", help="The string used to separate values.")
    parser.add_argument("--path-to-output-file", default="gnn_fiedler_approx/custom_models/embeddings/karate3.txt", help="Path to output file")
    parser.add_argument("--radius", type=int, default=2, help="Maximum radius of ego-networks.")
    parser.add_argument("--dim", type=int, default=10, help="Dimensionality of the embeddings.")
    parser.add_argument("--kernel", default="shortest_path", help="Graph kernel (shortest_path or weisfeiler_lehman).")
    args = parser.parse_args()

    np.random.seed(42)

    dataset = KarateClub()
    data = dataset[0]  # Get the first graph in the dataset

    segk = SEGK(radius=args.radius, dim=args.dim, kernel=args.kernel)
    data = segk(data)

    segk.write_to_file(args.path_to_output_file)


if __name__ == "__main__":
    main()
