import torch
from torch import nn
from torch.nn import Parameter
import math
from .base_layers import Activation



class GCNLayer(nn.Module):
    def __init__(self,  hidden_dim: int, dropout: float, activation: str, bias: bool):
        """
        Default Graph Convolutional Network layer class.

        :param hidden_dim: int: The input dimension of the layer.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(GCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(hidden_dim))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)
        self.activation = Activation(activation)
        self.reset_parameters()  # Initialize the parameters

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, last_layer: bool = False):
        """
        Forward pass of the GCN layer.

        :param h: torch.Tensor: The input node embeddings.
        :param adj: torch.Tensor: The adjacency matrix.
        :param last_layer: bool: Whether this is the last layer of the network.
        """
        # Perform the GCN operation
        support = torch.mm(h, self.weight)
        output = torch.spmm(adj, support)

        # Add bias if needed
        if self.bias is not None:
            output = output + self.bias

        # Apply activation and dropout if needed
        output = self.activation(output)
        if not last_layer:
            output = self.dropout(output)

        return output


class GCNModel(nn.Module):
    def __init__(self, node_in_dim: int, node_out_dim: int = 1, hidden_dim: int = 128, n_layers: int = 3,
                 dropout: float = 0.0,  activation: str = 'relu', bias: bool = True):
        """
        Default Graph Convolutional Network model class.

        :param node_in_dim: int: The input dimension of the nodes.
        :param node_out_dim: int: The output dimension of the nodes.
        :param hidden_dim: int: The hidden dimension of the network.
        :param n_layers: int: The number of layers in the network.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(GCNModel, self).__init__()
        self.in_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, dropout, activation, bias) for _ in range(n_layers)])
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
            h = layer(h, state.adj_matrix, last_layer=(idx == len(self.layers) - 1))

        # Final projection to the output action logits
        return self.out_projection(h)
