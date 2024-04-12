import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .layers import GTLayer, EdgeGTLayer



class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *inputs):
        pass


class GTModel(BaseModel):
    def __init__(self, node_in_dim=1, node_out_dim=1, hidden_dim=128, n_layers=3, mult_hidden=4, n_heads=8, dropout=0.0, activation='relu',
                 normalization='layer', bias=False):
        super(GTModel, self).__init__()
        self.in_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([GTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(hidden_dim, node_out_dim, bias=bias)

    def forward(self, state):
        h = self.in_projection(state.node_features)
        for layer in self.layers:
            h = layer(h)
        return self.out_projection(h)



class EdgeInGTModel(BaseModel):
    def __init__(self, node_in_dim=1, node_out_dim=1, edge_in_dim=1, hidden_dim=128, n_layers=3, mult_hidden=4,
                 n_heads=8, dropout=0.0, activation='relu', normalization='layer', bias=False):
        super(EdgeInGTModel, self).__init__()
        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.in_edge_projection = nn.Linear(edge_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([EdgeGTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(hidden_dim, node_out_dim, bias=bias)

    def forward(self, state):
        h = self.in_node_projection(state.node_features)
        e = self.in_edge_projection(state.edge_features)
        for layer in self.layers:
            h = layer(h, e)
        return self.out_projection(h)


class EdgeOutGTModel(BaseModel):
    def __init__(self, node_in_dim, edge_out_dim, hidden_dim=128, n_layers=3, mult_hidden=4,
                 n_heads=8, dropout=0.0, activation='relu', normalization='layer', bias=False):
        super(EdgeOutGTModel, self).__init__()
        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([EdgeGTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(hidden_dim, edge_out_dim, bias=bias)

    def forward(self, state):
        h = self.in_node_projection(state.node_features)
        for layer in self.layers:
            h = layer(h)
        # obtain e_ij by concatenating h_i and h_j
        e_out = torch.cat([h.unsqueeze(1).expand(-1, state.problem_size, -1, -1), h.unsqueeze(2).expand(-1, -1, state.problem_size, -1)], dim=-1)
        logits = self.out_projection(e_out)
        return logits.reshape(state.batch_size, state.problem_size*state.problem_size, -1)


class EdgeInOutGTModel(BaseModel):
    def __init__(self, node_in_dim, edge_in_dim, edge_out_dim, hidden_dim=128, n_layers=3, mult_hidden=4, n_heads=8, dropout=0.0, activation='relu',
                 normalization='layer', bias=False):
        super(EdgeInOutGTModel, self).__init__()
        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.in_edge_projection = nn.Linear(edge_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([EdgeGTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(2*hidden_dim, edge_out_dim, bias=bias)

    def forward(self, state):
        h = self.in_node_projection(state.node_features)
        e = self.in_edge_projection(state.edge_features)
        for layer in self.layers:
            h = layer(h, e)
        # obtain e_ij by concatenating h_i and h_j
        e_out = torch.cat([h.unsqueeze(1).expand(-1, state.problem_size, -1, -1), h.unsqueeze(2).expand(-1, -1, state.problem_size, -1)], dim=-1)
        logits = self.out_projection(e_out)
        return logits.reshape(state.batch_size, state.problem_size*state.problem_size, -1)


