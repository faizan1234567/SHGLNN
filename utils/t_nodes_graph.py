import torch_geometric
import numpy as np
from get_dataset import Dataset

def total_nodes_in_a_graph(dataset_name, save_dir = None):
    nodes = 0
    if not dataset_name.isupper():
        dataset_name = dataset_name.upper()
    data_instance = Dataset(name= dataset_name, save_dir= save_dir)
    data = data_instance.return_dataset()
    total_graphs = len(data)
    for i in range(total_graphs):
        sub_graph_nodes = data[i].num_nodes
        nodes += sub_graph_nodes
    return nodes, data

if __name__ == "__main__":
   name = 'mutag'
   nodes = total_nodes_in_a_graph(f"{name}", f"../datasets/{name}")
   print(f'total nodes in the dataset: {nodes}')
