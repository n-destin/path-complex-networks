import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, TransformerConv, GATConv, GCNConv
from lib.layers.non_linear import get_nonlinearity
from lib.layers.pooling import get_pooling_fn
from models.graph_models.helpers import block, reset_parameters, post_layers, apply_block
from lib.layers.norm import get_graph_norm


class GraphNN(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, readout='sum',
                 dropout_rate=0.5, nonlinearity='relu', heads = 1, task = None, graph_norm = "ln", type = "graphsage"):
        super(GraphNN, self).__init__()
        self.pooling_fn = get_pooling_fn(readout) 
        self.nonlinearity_type = nonlinearity
        self.nonlinearity = get_nonlinearity(nonlinearity, return_module=False)
        self.dropout_rate = dropout_rate
        self.task = task
        self.hidden = hidden
        self.heads = heads

        pair_fator = 2 if self.task == "pair" else 1

        if type in {"graphsage", "gcn"} and heads != 1:
            raise ValueError(f"{type} does not support heads={heads} in this implementation")

        ConvLayer = None
        ConvLayer = SAGEConv if type == "graphsage" else TransformerConv if type == "graphTransformer" else GATConv if type == "GAT" else GCNConv if type == "gcn" else None 
        
        if ConvLayer is None: raise ValueError('Invalid GNN Type {}'.format(type))

        self.conv1 = ConvLayer(num_features, hidden) if self.heads == 1 else ConvLayer(num_features, hidden, self.heads)
        if self.heads > 1: self.linear1 = Linear(self.heads * hidden, hidden)
        self.norm1 = get_graph_norm(graph_norm)(num_features)
        self.convs, self.projs, self.layernorms1, self.layernorms2, self.mlps = block(self, ConvLayer, num_layers, graph_norm, hidden, hidden)
        
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(pair_fator * hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.norm1.reset_parameters()
        if self.heads > 1: self.linear1.reset_parameters()
        reset_parameters(self.convs, self.projs, self.layernorms1, self.layernorms2, self.mlps)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, train_ids = None, val_ids = None, test_ids = None):
        x, batch = apply_block(self, data, self.convs, self.layernorms1, self.layernorms2, self.mlps)
        return post_layers(self, x, batch=batch, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    def __repr__(self):
        return self.__class__.__name__