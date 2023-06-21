"""

Generate K matrix from the following datasets
- COLLAB
- IMDB
- PROTEINS
- MUTAG

The K matrix calculates the degree between the different
nodes in the graph such as source and target node
"""
import pandas as pd
import numpy as np
import torch_geometric
from get_dataset import Dataset
import networkx as nx


def tensor_to_list(tensor):
    '''
    :param tensor: type torch.Tensor

    :return: list
    '''
    array = tensor.numpy()
    array2list = list(array)
    return array2list

def get_src_target_nodes(graph):
    """
    Params:
    graph: torch_geometric.datasets.TUDataset
    return:
    source_nodes: list
    target_nodes: list
    """
    source_nodes = graph["edge_index"][0, :]
    target_nodes = graph["edge_index"][1, :]
    source_nodes = tensor_to_list(source_nodes)
    target_nodes = tensor_to_list(target_nodes)
    return source_nodes, target_nodes


def create_graph(graph, source_nodes, target_nodes):
    """
    Parameters
    ----------
    graph: torch_geometric
    source_nodes: list
    target_nodes: list

    Return
    ------
    spl: nx.Graph.shortest_path
    new_graph: nx.Graph
    k_matrix: np.ndarray
    """
    unique_nodes = list(np.unique(graph["edge_index"]))
    num_nodes = len(source_nodes)
    k_matrix = np.zeros((num_nodes, num_nodes))
    new_graph = nx.Graph()
    new_graph.add_nodes_from(unique_nodes)
    edge_info = list(zip(source_nodes, target_nodes))
    new_graph.add_edges_from(edge_info)
    spl = dict(nx.all_pairs_shortest_path_length(new_graph))
    return spl, new_graph, k_matrix


def calculateLength(a, b, spl):
    try:
        return spl[a][b]
    except KeyError:
        return 0


def generate_kmatrix(initial_k_matrix,
                     source_nodes,
                     target_nodes,
                     spl,
                     k_hops = None):
    kmatrix = initial_k_matrix
    if k_hops is not None:
        for i, row in enumerate(source_nodes):
            for j, col in enumerate(target_nodes):
                length = calculateLength(row, col, spl)
                if length <= k_hops and length != 0:
                    kmatrix[i, j] = 1
                else:
                    kmatrix[i, j] = 0
    return kmatrix

if __name__ == "__main__":
    NAME = "mutag".upper()
    dataset = Dataset(f"{NAME}", save_dir= f"../datasets/{NAME}")
    data = dataset.return_dataset()
    first_graph = data[0]
    src_nodes, target_nodes = get_src_target_nodes(first_graph)
    spl, new_graph, kmatrix = create_graph(first_graph, src_nodes, target_nodes)
    kmatrix = generate_kmatrix(kmatrix, src_nodes, target_nodes, spl, 2)
    print(f"K matrix generated: {kmatrix}")
    print(f'K matrix shape: {kmatrix.shape}')






