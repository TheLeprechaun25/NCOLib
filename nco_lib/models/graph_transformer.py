import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from .base_layers import MLP, Norm


class BaseGTLayer(nn.Module):
    def __init__(self, hidden_dim: int, mult_hidden: int, n_heads: int, dropout: float, activation: str,
                 normalization: str, bias: bool = False):
        """
        Base Graph Transformer layer class.
        :param hidden_dim: int: The input dimension of the layer.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        """
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
    def __init__(self, hidden_dim: int, mult_hidden: int, n_heads: int, dropout: float, activation: str,
                 normalization: str, bias: bool = False):
        super(GTLayer, self).__init__(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
        self.W_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)  # Linear transformation for q, k, v

    def forward(self, h):
        """
        Forward pass for the GTLayer.
        :param h: torch.Tensor: The node embeddings. Shape: (batch_size, n_nodes, hidden_dim).
        """
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
    def __init__(self, hidden_dim: int, mult_hidden: int, n_heads: int, dropout: float, activation: str,
                 normalization: str, bias: bool = False):
        super(EdgeGTLayer, self).__init__(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
        self.W_h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)  # Linear transformation for q, k, v
        self.W_e = nn.Linear(hidden_dim, 2 * n_heads, bias=bias)  # Additional edge weights

    def forward(self, h: torch.Tensor, e: torch.Tensor):
        """
        Forward pass for the EdgeGTLayer.
        :param h: torch.Tensor: The node embeddings. Shape: (batch_size, n_nodes, hidden_dim).
        :param e: torch.Tensor: The edge embeddings. Shape: (batch_size, n_nodes, n_nodes, hidden_dim).
        """
        batch_size, n_nodes, _ = h.shape
        h_in = h.clone()

        # Initial normalization
        h = self.norm1(h)

        # Linear transformation for node embeddings
        q, k, v = self.W_h(h).split(self.hidden_dim, dim=2)
        k = k.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Linear transformation for edge embeddings
        e1, e2 = self.W_e(e).split(self.n_heads, dim=3)
        e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
        e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)

        # Attention mechanism
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + e1  # Add edge weights to attention scores
        att = F.softmax(att, dim=-1)
        att = att * e2  # Multiply edge weights with attention scores
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(batch_size, n_nodes, self.hidden_dim)

        # Add residual, Normalization and MLP
        out = self.mlp(self.norm2(y + h_in))

        # Final residual connection
        return out + y


class BaseGTModel(nn.Module):
    def __init__(self):
        """
        Base Graph Transformer model class.
        """
        super(BaseGTModel, self).__init__()

    @abstractmethod
    def forward(self, *inputs):
        pass


class GTModel(BaseGTModel):
    def __init__(self, node_in_dim: int, node_out_dim: int = 1, hidden_dim: int = 128, n_layers: int = 3,
                 mult_hidden: int = 4, n_heads: int = 8, dropout: float = 0.0, activation: str = 'relu',
                 normalization: str = 'layer', bias: bool = False):
        """
        Node-based featured Graph Transformer model class with node-based action outputs.

        :param node_in_dim: int: The input dimension of the node features.
        :param node_out_dim: int: The output dimension of the node-based action logits.
        :param hidden_dim: int: The hidden dimension of the model.
        :param n_layers: int: The number of layers in the model.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(GTModel, self).__init__()
        self.in_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([GTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(hidden_dim, node_out_dim, bias=bias)

    def forward(self, state):
        # Initial projection from node features to node embeddings
        h = self.in_projection(state.node_features)

        # Pass through the layers
        for layer in self.layers:
            h = layer(h)

        # Final projection to node-based action logits
        return self.out_projection(h)


class EdgeInGTModel(BaseGTModel):
    def __init__(self, node_in_dim: int, edge_in_dim: int, node_out_dim: int = 1, hidden_dim: int = 128,
                 n_layers: int = 3, mult_hidden: int = 4, n_heads: int = 8, dropout: float = 0.0,
                 activation: str = 'relu', normalization: str = 'layer', bias: bool = False):
        """
        Node- and Edge-based featured Graph Transformer model class with node-based action outputs.

        :param node_in_dim: int: The input dimension of the node features.
        :param edge_in_dim: int: The input dimension of the edge features.
        :param node_out_dim: int: The output dimension of the node-based action logits.
        :param hidden_dim: int: The hidden dimension of the model.
        :param n_layers: int: The number of layers in the model.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(EdgeInGTModel, self).__init__()
        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.in_edge_projection = nn.Linear(edge_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([EdgeGTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(hidden_dim, node_out_dim, bias=bias)

    def forward(self, state):
        # Initial projection from node features to node embeddings
        h = self.in_node_projection(state.node_features)

        # Initial projection from edge features to edge embeddings
        e = self.in_edge_projection(state.edge_features)

        # Pass through the layers
        for layer in self.layers:
            h = layer(h, e)

        # Final projection to node-based action logits
        return self.out_projection(h)


class EdgeOutGTModel(BaseGTModel):
    def __init__(self, node_in_dim: int, edge_out_dim: int = 1, hidden_dim: int = 128, n_layers: int = 3,
                 mult_hidden: int = 4, n_heads: int = 8, dropout: float = 0.0, activation: str = 'relu',
                 normalization: str = 'layer', bias: bool = False):
        """
        Node-based Graph Transformer model class with edge-based action outputs.

        :param node_in_dim: int: The input dimension of the node features.
        :param edge_out_dim: int: The output dimension of the edge-based action logits.
        :param hidden_dim: int: The hidden dimension of the model.
        :param n_layers: int: The number of layers in the model.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(EdgeOutGTModel, self).__init__()
        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([EdgeGTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(hidden_dim, edge_out_dim, bias=bias)

    def forward(self, state):
        # Initial projection from node features to node embeddings
        h = self.in_node_projection(state.node_features)

        # Pass through the layers
        for layer in self.layers:
            h = layer(h)

        # Obtain edge embedding e_ij by concatenating h_i and h_j
        e_out = torch.cat([h.unsqueeze(1).expand(-1, state.problem_size, -1, -1), h.unsqueeze(2).expand(-1, -1, state.problem_size, -1)], dim=-1)

        # Final projection to edge-based action logits
        logits = self.out_projection(e_out)
        return logits.reshape(state.batch_size, state.problem_size*state.problem_size, -1)


class EdgeInOutGTModel(BaseGTModel):
    def __init__(self, node_in_dim: int, edge_in_dim: int, edge_out_dim: int = 1, hidden_dim: int = 128,
                 n_layers: int = 3, mult_hidden: int = 4, n_heads: int = 8, dropout: float = 0.0,
                 activation: str = 'relu', normalization: str = 'layer', bias: bool = False):
        """
        Node- and Edge-based Graph Transformer model class with edge-based action outputs.

        :param node_in_dim: int: The input dimension of the node features.
        :param edge_in_dim: int: The input dimension of the edge features.
        :param edge_out_dim: int: The output dimension of the edge-based action logits.
        :param hidden_dim: int: The hidden dimension of the model.
        :param n_layers: int: The number of layers in the model.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(EdgeInOutGTModel, self).__init__()
        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.in_edge_projection = nn.Linear(edge_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([EdgeGTLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                     for _ in range(n_layers)])
        self.out_projection = nn.Linear(2*hidden_dim, edge_out_dim, bias=bias)

    def forward(self, state):
        # Initial projection from node features to node embeddings
        h = self.in_node_projection(state.node_features)

        # Initial projection from edge features to edge embeddings
        e = self.in_edge_projection(state.edge_features)

        # Pass through the layers
        for layer in self.layers:
            h = layer(h, e)

        # Obtain edge embedding e_ij by concatenating h_i and h_j
        e_out = torch.cat([h.unsqueeze(1).expand(-1, state.problem_size, -1, -1), h.unsqueeze(2).expand(-1, -1, state.problem_size, -1)], dim=-1)

        # Final projection to edge-based action logits
        logits = self.out_projection(e_out)
        return logits.reshape(state.batch_size, state.problem_size*state.problem_size, -1)
