import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
from .base_layers import Activation
from .decoders import DECODER_DICT


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetrically normalize a batch of adjacency matrices: \hat{A} = D^{-1/2}(A + I)D^{-1/2}
    :param adj: Tensor of shape (B, N, N)
    :return: Tensor of shape (B, N, N)
    """
    I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0).expand_as(adj)
    A = adj + I
    deg = A.sum(dim=-1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        activation: str = 'relu',
        bias: bool = True,
        dropout: float = 0.0,
        edge_dropout: float = 0.0,
        residual: bool = True,
        batch_norm: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.activation = Activation(activation)
        self.dropout_feat = nn.Dropout(dropout) if dropout > 0 else None
        self.edge_dropout = nn.Dropout(edge_dropout) if edge_dropout > 0 else None
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        self.residual = residual
        if self.residual:
            self.res_weight = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        if self.batch_norm is not None:
            nn.init.ones_(self.batch_norm.weight)
            nn.init.zeros_(self.batch_norm.bias)
        if self.residual:
            nn.init.constant_(self.res_weight, 0.0)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Optionally drop random edges
        if self.edge_dropout is not None:
            adj = self.edge_dropout(adj)
        # Normalize adjacency
        A_hat = normalize_adjacency(adj)
        # Linear transform
        h_lin = self.linear(h)
        # Convolution / aggregation
        out = torch.bmm(A_hat, h_lin)
        # BatchNorm
        if self.batch_norm is not None:
            B, N, _ = out.size()
            out = out.view(B * N, self.hidden_dim)
            out = self.batch_norm(out)
            out = out.view(B, N, self.hidden_dim)
        # Activation
        out = self.activation(out)
        # Feature dropout
        if self.dropout_feat is not None:
            out = self.dropout_feat(out)
        # Residual connection
        if self.residual:
            out = out + self.res_weight * h
        return out


class GCNModel(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        node_out_dim: int = 1,
        decoder: str = 'linear',
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        edge_dropout: float = 0.0,
        activation: str = 'relu',
        bias: bool = True,
        batch_norm: bool = True,
        residual: bool = True,
        aux_node: bool = False,
        logit_clipping: float = 10.0,
        n_heads: int = 8
    ):
        super().__init__()
        self.out_dim = node_out_dim
        self.clip_logits = logit_clipping > 0.0
        self.logit_clipping = logit_clipping
        self.sqrt_hidden = math.sqrt(hidden_dim)

        # Initial node feature projection
        self.in_proj = nn.Linear(node_in_dim, hidden_dim, bias=bias)

        # Optional virtual (global) node
        self.aux_node = aux_node
        if aux_node:
            # project pooled summary into hidden space
            self.aux_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # GCN layers
        self.layers = nn.ModuleList([
            GCNLayer(
                hidden_dim=hidden_dim,
                activation=activation,
                bias=bias,
                dropout=dropout,
                edge_dropout=edge_dropout,
                residual=residual,
                batch_norm=batch_norm
            ) for _ in range(n_layers)
        ])

        # Final decoder
        assert decoder in DECODER_DICT, f"Decoder must be one of {list(DECODER_DICT.keys())}"
        self.decoder = DECODER_DICT[decoder](hidden_dim, node_out_dim, n_heads, aux_node, bias=bias)

    def forward(self, state):
        # state.node_features: (BATCH, pomo, N, node_in_dim)
        h = self.in_proj(state.node_features.clone().view(-1, state.node_features.size(2), state.node_features.size(3)))  # (B, p, N, hidden_dim)

        # Append virtual node if requested
        if self.aux_node:
            # compute graph-level summary
            summary = h.mean(dim=1, keepdim=True)           # (Bp, 1, hidden_dim)
            v = self.aux_proj(summary)                      # (Bp, 1, hidden_dim)
            h = torch.cat([h, v], dim=1)             # (Bp, N+1, hidden_dim)

            adj = state.adj_matrix
            virtual_edges = torch.ones(state.batch_size*state.pomo_size, 1, state.problem_size,  dtype=torch.long, device=state.device)
            adj = torch.cat([adj, virtual_edges], dim=1)
            virtual_edges_t = torch.ones(state.batch_size*state.pomo_size, state.problem_size + 1, 1, dtype=torch.long, device=state.device)
            adj = torch.cat([adj, virtual_edges_t], dim=2)

        else:
            adj = state.adj_matrix

        # Pass through GCN layers
        for layer in self.layers:
            h = layer(h, adj)

        # Decode to logits
        out = self.decoder(h)

        # Logit clipping/stabilization
        if self.clip_logits:
            out = out / self.sqrt_hidden
            out = self.logit_clipping * torch.tanh(out)

        return out
