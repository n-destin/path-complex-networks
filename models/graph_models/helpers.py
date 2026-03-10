import torch
import torch.nn.functional as F
from torch.nn import Linear
from lib.layers.non_linear import get_nonlinearity
from lib.layers.norm import get_graph_norm


def apply_block(self, data, *layers):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    x = self.conv1((self.norm1(x.float())), edge_index)
    if self.heads > 1:
        assert hasattr(self, "linear1"), "Error: linear1 not initalized in graphTransformer"
        x = self.linear1(x)

    x = self.nonlinearity(x)

    for i, (conv, norm1, norm2, mlp) in enumerate(zip(*layers)):
            h = x
            x = conv(norm1(x), edge_index)
            if self.heads and self.heads > 1:
                x = self.projs[i](x)
            x = self.nonlinearity(x)
            x = h + F.dropout(x, p=self.dropout_rate, training=self.training)

            h = x 
            x = norm2(x)
            x = h + F.dropout(mlp(x), p = self.dropout_rate, training = self.training)

    if self.task != "pair":
        x = self.pooling_fn(x, batch)
    
    return x, batch


def reset_parameters(*block):
    for elem in block:
        for layer in elem:
            if isinstance(layer, torch.nn.ModuleList):
                reset_parameters(*layer)
                continue
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


def block(self, Layer, num_layers, graph_norm, *args, **kwargs):
    LayerNorm = get_graph_norm(graph_norm)
    if self.heads > 1: args += (self.heads,)
    convs = torch.nn.ModuleList(
        [Layer(*args, **kwargs) for _ in range(num_layers - 1)])
    projs = torch.nn.ModuleList()
    if self.heads and self.heads > 1:
         projs = torch.nn.ModuleList(
            [Linear(self.heads * self.hidden, self.hidden) for _ in range(num_layers - 1)
        ])
    mlps = torch.nn.ModuleList(
        [return_mlp(self, 4) for _ in range(num_layers - 1)]
    )
    layer_norms1 = torch.nn.ModuleList(
        [LayerNorm(self.hidden) for _ in range(num_layers - 1)]
    )
    layer_norms2 = torch.nn.ModuleList(
        [LayerNorm(self.hidden) for _ in range(num_layers - 1)]
    )

    return convs, projs, layer_norms1, layer_norms2, mlps


def post_layers(self, x, batch, train_ids = None, val_ids = None, test_ids = None):
    x = self.nonlinearity(self.lin1(x))
    x = F.dropout(x, p=self.dropout_rate, training=self.training)

    if self.training:
        ids = train_ids 
    elif self.eval:
        assert val_ids or test_ids
        ids = val_ids if val_ids else test_ids

    if self.task == "pair":
        outputs = []
        num_graphs = int(batch.max().item() + 1)
        for g in range(num_graphs):
            mask = (batch == g)
            xb = x[mask]
            pair_ids = torch.as_tensor(ids, device=xb.device, dtype=torch.long)
            
            print(type(ids), "we're here")
            
            if pair_ids.numel() == 0:
                continue
            if pair_ids.dim() == 1:
                pair_ids = pair_ids.view(-1, 2)

            a, b = pair_ids[:, 0], pair_ids[:, 1]
            pair_feat = torch.cat([xb[a], xb[b]], dim=1)
            logits = self.lin2(pair_feat)
            outputs.append(logits)
        outputs = torch.cat(outputs, dim = 0)
        return outputs

    x = self.lin2(x)
    return x


def return_mlp(self, hidden_factor):
    return torch.nn.Sequential(
        Linear(self.hidden, self.hidden * hidden_factor),
        get_nonlinearity(self.nonlinearity_type)(), 
        torch.nn.Dropout(self.dropout_rate),
        Linear(hidden_factor * self.hidden, self.hidden),
        torch.nn.Dropout(self.dropout_rate)
    )