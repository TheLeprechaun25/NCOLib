import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_layers import MLP, Norm
from .decoders import DECODER_DICT


class EdgeGNNEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, mult_hidden: int, n_heads: int, dropout: float, activation: str,
                 normalization: str, bias: bool = False):
        super(EdgeGNNEncoderLayer, self).__init__()
        """
        Edge GNN layer class.
        :param hidden_dim: int: The input dimension of the layer.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = dropout

        self.norm_h = Norm(hidden_dim=hidden_dim, normalization=normalization)
        self.norm_e = Norm(hidden_dim=hidden_dim, normalization=normalization)
        self.mlp = MLP(hidden_dim=hidden_dim, mult_hidden=mult_hidden, activation=activation, dropout=dropout, bias=bias)

        self.U = nn.Linear(self.hidden_dim,  self.hidden_dim, bias=bias)
        self.V = nn.Linear(self.hidden_dim,  self.hidden_dim, bias=bias)
        self.A = nn.Linear(self.hidden_dim,  self.hidden_dim, bias=bias)
        self.B = nn.Linear(self.hidden_dim,  self.hidden_dim, bias=bias)
        self.C = nn.Linear(self.hidden_dim,  self.hidden_dim, bias=bias)

    def forward(self, h, e):
        """
        Forward pass for the EdgeGNNEncoderLayer.
        :param h: torch.Tensor: The node embeddings. Shape: (batch_size, n_nodes, hidden_dim).
        :param e: torch.Tensor: The edge embeddings. Shape: (batch_size, n_nodes, n_nodes, hidden_dim).
        """
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        Vh = gates * Vh
        h = h + torch.sum(Vh, dim=2)

        # Normalize node features
        h = self.norm_h(h.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)

        # Normalize edge features
        e = self.norm_e(e.view(batch_size * num_nodes * num_nodes, hidden_dim)).view(batch_size, num_nodes, num_nodes, hidden_dim)

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e

        return h, e



class EdgeGNNModel(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, edge_out_dim: int = 1, decoder: str = 'linear', hidden_dim: int = 128, n_encoder_layers: int = 3,
                 mult_hidden: int = 4, n_heads: int = 8, dropout: float = 0.0, activation: str = 'relu',
                 normalization: str = 'layer', bias: bool = False, aux_node: bool = False, logit_clipping: float = 0.0):
        """
        :param node_in_dim: int: The input dimension of the node features.
        :param edge_in_dim: int: The input dimension of the edge features.
        :param edge_out_dim: int: The output dimension of the edge-based action logits.
        :param decoder: str: The decoder to use. Options: 'linear', 'attention'.
        :param hidden_dim: int: The hidden dimension of the model.
        :param n_encoder_layers: int: The number of layers in the model.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        :param bias: bool: Whether to use bias in the linear layers.
        :param aux_node: bool: Whether to use an auxiliary (also known as virtual) node.
        :param logit_clipping: float: The logit clipping value. 0.0 means no clipping. 10.0 is a commonly used value.
        """
        super(EdgeGNNModel, self).__init__()
        self.out_dim = edge_out_dim
        self.sqrt_embedding_dim = math.sqrt(hidden_dim)

        self.clip_logits = logit_clipping > 0.0
        self.logit_clipping = logit_clipping

        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.in_edge_projection = nn.Linear(edge_in_dim, hidden_dim, bias=bias)
        self.encoder_layers = nn.ModuleList([EdgeGNNEncoderLayer(hidden_dim, mult_hidden, n_heads, dropout, activation,
                                                                 normalization, bias) for _ in range(n_encoder_layers)])

        assert decoder in DECODER_DICT.keys(), f"Decoder must be one of {DECODER_DICT.keys()}"
        self.decoder = DECODER_DICT[decoder](hidden_dim, edge_out_dim, n_heads, aux_node, bias=bias)

    def forward(self, state):
        """
        Forward pass for the GTModel.
        :param state: State: The state of the environment.
        """

        # Reshape the node features to (batch_size * pomo_size, n_nodes, features)
        node_features = state.node_features.clone().view(-1, state.node_features.size(2), state.node_features.size(3))

        # Initial projection from node features to node embeddings
        h = self.in_node_projection(node_features)

        # Edge features
        edge_feat = state.edge_features.clone().view(state.batch_size*state.pomo_size, state.problem_size, state.problem_size, -1)

        # Initial projection from edge features to edge embeddings
        e = self.in_edge_projection(edge_feat)

        # Pass through the encoding layers
        for layer in self.encoder_layers:
            h, e = layer(h, e)

        # Reshape the edge embeddings to (batch_size * pomo_size, n_edges, hidden_dim)
        e = e.view(state.batch_size*state.pomo_size, state.problem_size * state.problem_size, -1)

        # Decode to edge-based action logits
        out, aux_node = self.decoder(e)

        if self.clip_logits:
            out = out / self.sqrt_embedding_dim
            out = self.logit_clipping * torch.tanh(out)

        return out, aux_node

