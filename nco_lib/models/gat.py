import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_layers import Activation



class GATLayer(nn.Module):
    def __init__(self,  hidden_dim: int, n_heads: int, is_concat: bool, leaky_relu_negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = False):
        """
        Default Graph Attention Network layer class.

        :param hidden_dim: int: The input dimension of the layer.
        :param n_heads: int: The number of attention heads.
        :param is_concat: bool: Whether to concatenate the attention heads or average them.
        :param leaky_relu_negative_slope: float: The negative slope of the LeakyReLU activation function.
        :param dropout: float: The dropout rate.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(GATLayer, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert hidden_dim % n_heads == 0
            self.n_hidden = hidden_dim // n_heads
        else:
            self.n_hidden = hidden_dim

        self.linear = nn.Linear(hidden_dim, self.n_hidden * n_heads, bias=bias)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=bias)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """
        Forward pass of the GCN layer.

        :param h: torch.Tensor: The input node embeddings.
        :param adj: torch.Tensor: The adjacency matrix.
        """
        batch_size, n_nodes = h.shape[:2]
        g = self.linear(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(1, n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(batch_size, n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat)).squeeze(-1)

        e = e.masked_fill(adj == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)
        if self.is_concat:
            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=2)


class GATV2Layer(nn.Module):
    def __init__(self,  hidden_dim: int, n_heads: int, is_concat: bool, leaky_relu_negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = False):
        """
        Default Graph Attention Network layer class.

        :param hidden_dim: int: The input dimension of the layer.
        :param n_heads: int: The number of attention heads.
        :param is_concat: bool: Whether to concatenate the attention heads or average them.
        :param leaky_relu_negative_slope: float: The negative slope of the LeakyReLU activation function.
        :param dropout: float: The dropout rate.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(GATV2Layer, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert hidden_dim % n_heads == 0
            self.n_hidden = hidden_dim // n_heads
        else:
            self.n_hidden = hidden_dim

        self.linear_l = nn.Linear(hidden_dim, self.n_hidden * n_heads, bias=bias)
        self.linear_r = nn.Linear(hidden_dim, self.n_hidden * n_heads, bias=bias)

        self.attn = nn.Linear(self.n_hidden, 1, bias=bias)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """
        Forward pass of the GCN layer.

        :param h: torch.Tensor: The input node embeddings.
        :param adj: torch.Tensor: The adjacency matrix.
        """
        batch_size, n_nodes = h.shape[:2]
        g_l = self.linear_l(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)

        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1)

        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1)

        g_sum = g_l_repeat + g_r_repeat_interleave

        g_sum = g_sum.view(batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum)).squeeze(-1)

        e = e.masked_fill(adj == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum('bijh,bjhf->bihf', a, g_r)
        if self.is_concat:
            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=2)


class GATModel(nn.Module):
    def __init__(self, node_in_dim: int, version: str = 'v2', node_out_dim: int = 1, hidden_dim: int = 128, n_layers: int = 3,
                 n_heads: int = 8, is_concat: bool = True, leaky_relu_negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = False):
        """
        Default Graph Convolutional Network model class.

        :param node_in_dim: int: The input dimension of the nodes.
        :param node_out_dim: int: The output dimension of the nodes.
        :param hidden_dim: int: The hidden dimension of the network.
        :param n_layers: int: The number of layers in the network.
        :param n_heads: int: The number of attention heads.
        :param is_concat: bool: Whether to concatenate the attention heads or average them.
        :param leaky_relu_negative_slope: float: The negative slope of the LeakyReLU activation function.
        :param dropout: float: The dropout rate.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(GATModel, self).__init__()
        self.in_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        if version == 'v1':
            self.layers = nn.ModuleList([GATLayer(hidden_dim, n_heads, is_concat, leaky_relu_negative_slope,  dropout, bias) for _ in range(n_layers)])
        elif version == 'v2':
            self.layers = nn.ModuleList([GATV2Layer(hidden_dim, n_heads, is_concat, leaky_relu_negative_slope,  dropout, bias) for _ in range(n_layers)])
        else:
            raise NotImplementedError

        self.out_projection = nn.Linear(hidden_dim, node_out_dim, bias=bias)


    def forward(self, state):
        """
        Forward pass of the GCN model.

        :param state: State: The state of the environment.
        """
        # Initial projection from node features to node embeddings
        h = self.in_projection(state.node_features)

        # Perform the forward pass through all layers
        for idx, layer in enumerate(self.layers):
            h = layer(h, state.adj_matrix.unsqueeze(-1))

        # Final projection to the output action logits
        return self.out_projection(h)
