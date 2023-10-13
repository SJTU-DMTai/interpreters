import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl
from .graph_model import GraphModel
from dgl.nn.pytorch import GraphConv


class GCNModel(GraphModel):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="LSTM",
                 num_graph_layer=2, heads=None, use_residual=False):
        super().__init__(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                         base_model=base_model, use_residual=use_residual, is_homogeneous=True)
        self.num_graph_layer = num_graph_layer
        self.gat_layers = nn.ModuleList()

        for i in range(num_graph_layer-1):
            self.gat_layers.append(
                GraphConv(hidden_size, hidden_size, activation=F.relu, allow_zero_in_degree=True, weight=False),
            )

        self.gat_layers.append(
            GraphConv(hidden_size, hidden_size, activation=None, allow_zero_in_degree=True, weight=False),
        )
        self.reset_parameters()
        for layer in self.gat_layers:
            layer._allow_zero_in_degree = True
        self._allow_zero_in_degree = True

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_out.weight, gain=gain)

    def forward_graph(self, h, index=None, return_subgraph=False, edge_weight=None):
        if index:
            subgraph = dgl.node_subgraph(self.g, index)
        else:
            subgraph = self.g
        for i, layer in enumerate(self.gat_layers):
            h = layer(subgraph, h, edge_weight=edge_weight)
        if return_subgraph:
            return h, subgraph
        else: return h
