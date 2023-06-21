""""
Get the five different types of the datasets, such as
- MUTAG
- COLLAB
- IMDB-BIN
- PROTEINS

These dataset are imported from TU dataset class from pytorch geometric"""
import torch_geometric
from  torch_geometric.datasets import TUDataset

class Dataset:
    """
    get the desired the based on the need
    """
    def __init__(self, name, graph = None,
                 print_info = False,
                 save_dir = "",
                 pre_transform = None,
                 post_transform = None):
        self.name = name
        self.graph = graph
        self.print_info = print_info
        self.save_dir = save_dir
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def return_dataset(self):
        """
        return required dataset
        """
        dataset = TUDataset(root = self.save_dir, name = self.name)
        if self.graph is None:
            return dataset
        else:
            return dataset[self.graph]

    def show_datast_info(self, data):
        """
        show the dataset information such as it's attributes
        Args:
        data: torch_geometric.dataset

        """
        print(f'key information: {data.keys}')
        print(f'number of nodes in the data: {data.num_nodes}')
        print(f'number of edges in the data: {data.num_edges}')  # using bidirectional edges...
        print(f'number of nodes features in the data: {data.num_node_features}')
        print(f'number of edge features in the data: {data.num_edge_features}')
        print(f'is the dataset isolated: {data.has_isolated_nodes()}')
        print(f'does the dataset has self loops: {data.has_self_loops()}')
        print(f'is the data directed: {data.is_directed()}')
        # print(f'Number of classes in the dataset: {data.num_classes}')


if __name__ == "__main__":
    # get proteins datasets for hypergraph generation...
    name = 'MUTAG'
    mutag = Dataset(f"{name}", save_dir= f"../datasets/{name}")
    data = mutag.return_dataset()
    mutag.show_datast_info(data[0])


