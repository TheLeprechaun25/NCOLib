import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from .base_layers import MLP, Norm


class BaseGTLayer(nn.Module):
    def __init__(self, hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias=False):
        super(BaseGTLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = dropout

        self.norm1 = Norm(hidden_dim=hidden_dim, normalization=normalization)
        self.norm2 = Norm(hidden_dim=hidden_dim, normalization=normalization)
        self.mlp = MLP(hidden_dim=hidden_dim, mult_hidden=mult_hidden, activation=activation, dropout=dropout, bias=bias)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented")


class GTLayer(BaseGTLayer):
    def __init__(self, hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias=False):
        super(GTLayer, self).__init__(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
        self.W_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)  # Linear transformation for q, k, v

    def forward(self, h):
        batch_size, n_nodes, _ = h.shape
        h_in = h.clone()

        # Initial normalization
        h = self.norm1(h)

        # Linear transformation
        q, k, v = self.W_h(h).split(self.hidden_dim, dim=2)
        k = k.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Attention mechanism
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(batch_size, n_nodes, self.hidden_dim)
        # all nan values are replaced with 0
        y = torch.where(torch.isnan(y), torch.zeros_like(y), y)

        # Add residual, Normalization and MLP
        out = self.mlp(self.norm2(y + h_in))

        # Final residual connection
        return out + y


class EdgeGTLayer(BaseGTLayer):
    def __init__(self, hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias=False):
        super(EdgeGTLayer, self).__init__(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
        self.W_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)  # Linear transformation for q, k, v
        self.W_e = nn.Linear(hidden_dim, 2 * n_heads, bias=bias)  # Additional edge weights

    def forward(self, h, e):
        batch_size, n_nodes, _ = h.shape
        h_in = h.clone()

        # Initial normalization
        h = self.norm1(h)

        # Linear transformation
        q, k, v = self.W_h(h).split(self.hidden_dim, dim=2)
        k = k.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        e1, e2 = self.W_e(e).split(self.n_heads, dim=3)
        e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
        e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)

        # Attention mechanism
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + e1
        att = F.softmax(att, dim=-1)
        att = att * e2
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(batch_size, n_nodes, self.hidden_dim)

        # Add residual, Normalization and MLP
        out = self.mlp(self.norm2(y + h_in))

        # Final residual connection
        return out + y


class BaseGTModel(nn.Module):
    def __init__(self):
        super(BaseGTModel, self).__init__()

    @abstractmethod
    def forward(self, *inputs):
        pass


class GTModel(BaseGTModel):
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



class EdgeInGTModel(BaseGTModel):
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


class EdgeOutGTModel(BaseGTModel):
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


class EdgeInOutGTModel(BaseGTModel):
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


