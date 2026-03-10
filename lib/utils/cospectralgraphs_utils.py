import networkx as nx 
import torch 
from torch_geometric.utils import to_undirected


def load_cospectral_graphs(path):
    nx_graphs = nx.read_graph6(path)
    graphs = list()
    for graph in nx_graphs:
        n = nx.number_of_nodes(graph)
        edge_index = to_undirected(torch.tensor(list(graph.edges()), dtype=torch.long).transpose(1, 0))
        graphs.append((edge_index, n))
    return graphs