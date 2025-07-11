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


def multi_head_attention(q, k, v):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed
