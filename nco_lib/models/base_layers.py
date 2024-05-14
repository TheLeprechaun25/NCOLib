import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mult_hidden: int, activation: str, dropout: float, bias: bool):
        """
        Default Multi-Layer Perceptron class with two linear layers, an activation function and dropout.
        :param hidden_dim: int: The input dimension of the MLP.
        :param mult_hidden: int: The multiplier for the hidden dimension of the first linear layer.
        :param activation: str: The activation function to use.
        :param dropout: float: The dropout rate.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super().__init__()
        self.c_fc = nn.Linear(hidden_dim, mult_hidden * hidden_dim, bias=bias)
        self.c_proj = nn.Linear(mult_hidden * hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = Activation(activation)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Norm(nn.Module):
    def __init__(self, hidden_dim: int, normalization: str):
        """
        Default normalization class.
        :param hidden_dim: int: The input dimension of the normalization.
        :param normalization: str: The normalization to use.
        """
        super().__init__()
        self.normalization = normalization

        if normalization == 'layer':
            self.norm = nn.LayerNorm(hidden_dim)
        elif normalization == 'batch':
            self.norm = nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False)
        elif normalization == 'instance':
            self.norm = nn.InstanceNorm1d(hidden_dim, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.normalization in ['instance', 'batch']:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm(x)
        return x


class Activation(nn.Module):
    def __init__(self, activation):
        """
        Default activation function class.
        :param activation: str: The activation function to use.
        """
        super().__init__()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)

