import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import math
from .base_layers import Activation



class GCNLayer(nn.Module):
    def __init__(self,  hidden_dim, dropout, activation, bias):
        super(GCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(hidden_dim))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)
        self.activation = Activation(activation)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, adj, last_layer=False):
        support = torch.mm(h, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if not last_layer:
            output = self.activation(output)
            output = self.dropout(output)

        return output


class GCN(nn.Module):
    def __init__(self, node_in_dim=1, node_out_dim=1, hidden_dim=128, n_layers=3, dropout=0.0,  activation='relu', bias=True):
        super(GCN, self).__init__()
        self.in_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, dropout, activation, bias) for _ in range(n_layers)])
        self.out_projection = nn.Linear(hidden_dim, node_out_dim, bias=bias)


    def forward(self, state):
        h = self.in_node_projection(state.node_features)
        adj = state.adj
        for idx, layer in enumerate(self.layers):
            h = layer(h, e)
        return self.out_projection(h)
