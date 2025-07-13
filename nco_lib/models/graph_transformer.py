import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from .base_layers import MLP, Norm
from .decoders import DECODER_DICT


class BaseGTEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, mult_hidden: int, n_heads: int, dropout: float, activation: str,
                 normalization: str, bias: bool = False, aux_node: bool = False):
        """
        Base Graph Transformer layer class.
        :param hidden_dim: int: The input dimension of the layer.
        :param mult_hidden: int: The multiplier for the hidden dimension of the MLP.
        :param n_heads: int: The number of attention heads.
        :param dropout: float: The dropout rate.
        :param activation: str: The activation function to use in the MLP.
        :param normalization: str: The normalization to use.
        :param bias: bool: Whether to use bias in the linear layers.
        :param aux_node: bool: Whether to use an auxiliary (also known as virtual) node.
        """
        super(BaseGTEncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = dropout
        self.aux_node = aux_node

        self.norm1 = Norm(hidden_dim=hidden_dim, normalization=normalization)
        self.norm2 = Norm(hidden_dim=hidden_dim, normalization=normalization)
        self.mlp = MLP(hidden_dim=hidden_dim, mult_hidden=mult_hidden, activation=activation, dropout=dropout, bias=bias)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented")


class GTEncoderLayer(BaseGTEncoderLayer):
    def __init__(self, hidden_dim: int, mult_hidden: int, n_heads: int, dropout: float, activation: str,
                 normalization: str, bias: bool = False):
        super(GTEncoderLayer, self).__init__(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
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


class EdgeGTEncoderLayer(BaseGTEncoderLayer):
    def __init__(self, hidden_dim: int, mult_hidden: int, n_heads: int, dropout: float, activation: str,
                 normalization: str, bias: bool = False):
        super(EdgeGTEncoderLayer, self).__init__(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization,
                                                 bias)
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
    def __init__(self, out_dim: int, hidden_dim: int, logit_clipping: float):
        """
        Base Graph Transformer model class.
        """
        super(BaseGTModel, self).__init__()
        self.out_dim = out_dim
        self.sqrt_embedding_dim = math.sqrt(hidden_dim)

        self.clip_logits = logit_clipping > 0.0
        self.logit_clipping = logit_clipping

    @abstractmethod
    def forward(self, *inputs):
        pass


class GTModel(BaseGTModel):
    def __init__(self, node_in_dim: int, node_out_dim: int = 1, decoder: str = 'linear', hidden_dim: int = 128, n_encoder_layers: int = 3,
                 mult_hidden: int = 4, n_heads: int = 8, dropout: float = 0.0, activation: str = 'relu',
                 normalization: str = 'layer', bias: bool = False, aux_node: bool = False, logit_clipping: float = 10.0):
        """
        Node-based featured Graph Transformer model class with node-based action outputs.

        :param node_in_dim: int: The input dimension of the node features.
        :param node_out_dim: int: The output dimension of the node-based action logits.
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
        super(GTModel, self).__init__(out_dim=node_out_dim, hidden_dim=hidden_dim, logit_clipping=logit_clipping)
        self.node_in_dim = node_in_dim
        self.node_out_dim = node_out_dim

        self.aux_node = aux_node
        if aux_node:
            self.virtual_nodes = nn.Parameter(torch.randn(node_out_dim, hidden_dim))

        self.in_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.encoder_layers = nn.ModuleList([GTEncoderLayer(hidden_dim, mult_hidden, n_heads, dropout, activation,
                                                            normalization, bias) for _ in range(n_encoder_layers)])

        assert decoder in DECODER_DICT.keys(), f"Decoder must be one of {DECODER_DICT.keys()}"
        self.decoder = DECODER_DICT[decoder](hidden_dim, node_out_dim, n_heads, aux_node, bias=bias)

    def forward(self, state):
        """
        Forward pass for the GTModel.
        :param state: State: The state of the environment.
        """

        # Reshape the node features to (batch_size * pomo_size, n_nodes, features)
        node_features = state.node_features.clone().view(-1, state.node_features.size(2), state.node_features.size(3))

        # Add memory information to node features
        if state.memory_info is not None:
            memory = state.memory_info.clone().view(state.batch_size*state.pomo_size, state.problem_size, -1)
            node_features = torch.cat([node_features, memory], dim=-1)

        # Initial projection from node features to node embeddings
        h = self.in_projection(node_features)

        if self.aux_node:
            # Append virtual node
            virtual_node_features = self.virtual_nodes.unsqueeze(0).repeat(h.size(0), 1, 1)

            h = torch.cat([h, virtual_node_features], dim=1)

        # Pass through the encoding layers
        for layer in self.encoder_layers:
            h = layer(h)

        # Decode to node-based action logits
        out = self.decoder(h)

        if self.clip_logits:
            out = out / self.sqrt_embedding_dim
            out = self.logit_clipping * torch.tanh(out)

        return out


class EdgeInGTModel(BaseGTModel):
    def __init__(self, node_in_dim: int, edge_in_dim: int, node_out_dim: int = 1, decoder: str = 'linear',
                 hidden_dim: int = 128, n_encoder_layers: int = 3, mult_hidden: int = 4, n_heads: int = 8,
                 dropout: float = 0.0, activation: str = 'relu', normalization: str = 'layer', bias: bool = False,
                 aux_node: bool = False, logit_clipping: float = 10.0):
        """
        Node- and Edge-based featured Graph Transformer model class with node-based action outputs.

        :param node_in_dim: int: The input dimension of the node features.
        :param edge_in_dim: int: The input dimension of the edge features.
        :param node_out_dim: int: The output dimension of the node-based action logits.
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
        super(EdgeInGTModel, self).__init__(out_dim=node_out_dim, hidden_dim=hidden_dim, logit_clipping=logit_clipping)
        self.node_in_dim = node_in_dim
        self.node_out_dim = node_out_dim
        self.edge_in_dim = edge_in_dim

        self.aux_node = aux_node
        if aux_node:
            self.virtual_nodes = nn.Parameter(torch.randn(1, hidden_dim))

        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.in_edge_projection = nn.Linear(edge_in_dim, hidden_dim, bias=bias)
        self.encoder_layers = nn.ModuleList([EdgeGTEncoderLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                            for _ in range(n_encoder_layers)])

        assert decoder in DECODER_DICT.keys(), f"Decoder must be one of {DECODER_DICT.keys()}"
        self.decoder = DECODER_DICT[decoder](hidden_dim, node_out_dim, n_heads, aux_node, bias=bias)

    def forward(self, state):
        # Reshape the node features to (batch_size * pomo_size, n_nodes, features)
        node_features = state.node_features.clone().view(state.batch_size*state.pomo_size, state.problem_size, -1)

        # Add memory information to node features
        if state.memory_info is not None:
            memory = state.memory_info.clone().view(state.batch_size*state.pomo_size, state.problem_size, -1)
            node_features = torch.cat([node_features, memory], dim=-1)

        # Initial projection from node features to node embeddings
        h = self.in_node_projection(node_features)

        # Edge features
        edge_feat = state.edge_features.clone().view(state.batch_size*state.pomo_size, state.problem_size, state.problem_size, -1)
        if self.aux_node:
            # Append virtual node
            virtual_node_features = self.virtual_nodes.unsqueeze(0).repeat(h.size(0), 1, 1)
            h = torch.cat([h, virtual_node_features], dim=1)

            # Update adjacency matrix for virtual node

            virtual_edges = torch.ones(state.batch_size*state.pomo_size, 1, state.problem_size, self.edge_in_dim,  dtype=torch.long, device=state.device)
            edge_feat = torch.cat([edge_feat, virtual_edges], dim=1)
            virtual_edges_t = torch.ones(state.batch_size*state.pomo_size, state.problem_size + 1, 1, self.edge_in_dim, dtype=torch.long, device=state.device)
            edge_feat = torch.cat([edge_feat, virtual_edges_t], dim=2)

        # Initial projection from edge features to edge embeddings
        e = self.in_edge_projection(edge_feat)

        # Pass through the layers
        for layer in self.encoder_layers:
            h = layer(h, e)

        # Decode to node-based action logits
        out = self.decoder(h)

        if self.clip_logits:
            out = out / self.sqrt_embedding_dim
            out = self.logit_clipping * torch.tanh(out)

        return out


class EdgeInOutGTModel(BaseGTModel):
    def __init__(self, node_in_dim: int, edge_in_dim: int, edge_out_dim: int = 1, decoder: str = 'edge',
                 hidden_dim: int = 128, n_encoder_layers: int = 3, mult_hidden: int = 4, n_heads: int = 8,
                 dropout: float = 0.0, activation: str = 'relu', normalization: str = 'layer', bias: bool = False,
                 aux_node: bool = False, logit_clipping: float = 10.0):
        """
        Node- and Edge-based Graph Transformer model class with edge-based action outputs.

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
        super(EdgeInOutGTModel, self).__init__(out_dim=edge_out_dim, hidden_dim=hidden_dim, logit_clipping=logit_clipping)
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.edge_out_dim = edge_out_dim
        self.aux_node = aux_node
        if aux_node:
            self.virtual_nodes = nn.Parameter(torch.randn(edge_out_dim, hidden_dim))

        self.in_node_projection = nn.Linear(node_in_dim, hidden_dim, bias=bias)
        self.in_edge_projection = nn.Linear(edge_in_dim, hidden_dim, bias=bias)
        self.encoder_layers = nn.ModuleList([EdgeGTEncoderLayer(hidden_dim, mult_hidden, n_heads, dropout, activation, normalization, bias)
                                             for _ in range(n_encoder_layers)])
        self.out_projection = nn.Linear(2*hidden_dim, edge_out_dim, bias=bias)
        assert decoder in DECODER_DICT.keys(), f"Decoder must be one of {DECODER_DICT.keys()}"
        if decoder in ['attention', 'linear']:
            print('Attention and Linear decoders are not supported for EdgeInOutGTModel. Using edge decoder instead.')
            decoder = 'edge'
        self.decoder = DECODER_DICT[decoder](2*hidden_dim, edge_out_dim, n_heads, aux_node, bias=bias)


    def forward(self, state):
        # Reshape the node features to (batch_size * pomo_size, n_nodes, features)
        node_features = state.node_features.clone().view(-1, state.node_features.size(2), state.node_features.size(3))

        # Add memory information to node features
        if state.memory_info is not None:
            memory = state.memory_info.clone().view(state.batch_size*state.pomo_size, state.problem_size, -1)
            node_features = torch.cat([node_features, memory], dim=-1)

        # Initial projection from node features to node embeddings
        h = self.in_node_projection(node_features)

        if self.aux_node:
            # Append virtual node
            virtual_node_features = self.virtual_nodes.unsqueeze(0).repeat(h.size(0), 1, 1)
            h = torch.cat([h, virtual_node_features], dim=1)

            # Update adjacency matrix for virtual node
            edge_feat = state.edge_features.clone()
            n_f = edge_feat.size(-1)
            virtual_edges = torch.ones(state.batch_size, state.pomo_size, 1, state.problem_size, n_f, dtype=torch.long, device=state.device)
            edge_feat = torch.cat([edge_feat, virtual_edges], dim=2)
            virtual_edges_t = torch.ones(state.batch_size, state.pomo_size, state.problem_size + 1, 1, n_f, dtype=torch.long, device=state.device)
            edge_feat = torch.cat([edge_feat, virtual_edges_t], dim=3)
        else:
            edge_feat = state.edge_features.clone()

        edge_feat = edge_feat.view(-1, edge_feat.size(2), edge_feat.size(3), edge_feat.size(4))

        # Initial projection from edge features to edge embeddings
        e = self.in_edge_projection(edge_feat)

        # Pass through the layers
        for layer in self.encoder_layers:
            h = layer(h, e)

        # Decode to edge-based action logits
        out = self.decoder(h)

        if self.clip_logits:
            out = out / self.sqrt_embedding_dim
            out = self.logit_clipping * torch.tanh(out)

        return out


class HypergraphConv(nn.Module):
    """
    Single layer of Hypergraph Convolution: node -> hyperedge -> node.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.edge_linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.node_linear = nn.Linear(out_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        x: [B*P, N, in_dim] node features
        H: [B*P, N, E] incidence matrix (binary or weighted)
        returns: [B*P, N, out_dim]
        """
        # 1) Node -> Hyperedge
        E = torch.bmm(H.transpose(1, 2), x)        # [B*P, E, in_dim]
        E = torch.relu(self.edge_linear(E))        # [B*P, E, out_dim]

        # 2) Hyperedge -> Node
        x2 = torch.bmm(H, E)                       # [B*P, N, out_dim]
        x2 = torch.relu(self.node_linear(x2))      # [B*P, N, out_dim]
        return x2


class HypergraphGNN(BaseGTModel):
    """
    Hypergraph Neural Network for node-level action scoring.
    Expects 'state.data["incidence"]' as [B, N, E] hyperedge incidence,
    and 'state.node_features' as [B, P, N, in_dim].
    """
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        node_out_dim: int = 1,
        decoder: str = "linear",
        mult_hidden: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = False,
        logit_clipping: float = 10.0,
    ):
        super().__init__(out_dim=node_out_dim,
                         hidden_dim=hidden_dim,
                         logit_clipping=logit_clipping)
        # initial node projection
        self.in_proj = nn.Linear(node_in_dim, hidden_dim, bias=bias)

        # hypergraph convolution layers
        self.hg_layers = nn.ModuleList([
            HypergraphConv(hidden_dim, hidden_dim, bias=bias)
            for _ in range(n_layers)
        ])

        # final fusion MLP + decoder
        self.mlp = MLP(hidden_dim, mult_hidden, activation, dropout, bias=bias)
        assert decoder in DECODER_DICT, f"Unknown decoder {decoder}"
        self.decoder = DECODER_DICT[decoder](hidden_dim, node_out_dim, 1, False, bias=bias)

    def forward(self, state):
        B, P, N = state.batch_size, state.pomo_size, state.problem_size
        # 1) reshape node features
        x = state.node_features.view(B * P, N, -1)   # [B*P, N, in_dim]
        x = self.in_proj(x)                          # [B*P, N, H]

        # 2) build hyperedge incidence per pomo
        H = state.data['incidence']                  # [B, N, E]
        H = H.unsqueeze(1).repeat(1, P, 1, 1)        # [B, P, N, E]
        H = H.view(B * P, N, -1)                     # [B*P, N, E]

        # 3) pass through hg layers
        for layer in self.hg_layers:
            x = layer(x, H)                          # [B*P, N, H]

        # 4) fuse (identity here) + decode
        h = self.mlp(x)                              # [B*P, N, H]
        out, aux = self.decoder(h)                   # [B*P, N, 1], aux

        out = out.view(B, P, N, -1)
        if self.clip_logits:
            out = out / self.sqrt_embedding_dim
            out = self.logit_clipping * torch.tanh(out)
        return out, aux
