import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self, ):
        """
        Base decoder class.
        """
        super(BaseDecoder, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented")


class LinearDecoder(BaseDecoder):
    def __init__(self, hidden_dim: int, node_out_dim: int, n_heads: int, aux_node: bool, bias: bool = False):
        """
        Linear decoder class.
        :param hidden_dim: int: The input dimension of the decoder.
        :param node_out_dim: int: The output dimension of the decoder.
        :param n_heads: int: The number of attention heads.
        :param aux_node: bool: Whether to use an auxiliary (also known as virtual) node.
        :param bias: bool: Whether to use bias in the linear layer.
        """
        super(LinearDecoder, self).__init__()
        self.node_out_dim = node_out_dim
        self.aux_node = aux_node
        self.linear = nn.Linear(hidden_dim, node_out_dim, bias=bias)

    def forward(self, h):
        """
        Forward pass for the LinearDecoder.
        :param h: torch.Tensor: The input embeddings. Shape: (batch_size, n_nodes or n_edges, hidden_dim).
        """
        if self.aux_node:
            # Divide aux node from the node embeddings
            h, h_g = h[:, :-self.node_out_dim, :], h[:, -1:, :]
        else:
            # Graph embedding isn't needed in linear decoder
            h_g = None

        return self.linear(h), h_g


class LinearEdgeDecoder(BaseDecoder):
    def __init__(self, hidden_dim: int, edge_out_dim: int, n_heads: int, aux_node: bool, bias: bool = False):
        """
        Linear decoder class.
        :param hidden_dim: int: The input dimension of the decoder.
        :param edge_out_dim: int: The output dimension of the decoder.
        :param n_heads: int: The number of attention heads.
        :param aux_node: bool: Whether to use an auxiliary (also known as virtual) node.
        :param bias: bool: Whether to use bias in the linear layer.
        """
        super(LinearEdgeDecoder, self).__init__()
        self.edge_out_dim = edge_out_dim
        self.aux_node = aux_node
        self.linear = nn.Linear(hidden_dim, edge_out_dim, bias=bias)

    def forward(self, e):
        """
        Forward pass for the LinearDecoder.
        :param e: torch.Tensor: The input edge embeddings. Shape: (batch_size, n_edges, hidden_dim).
        """
        return self.linear(e), None


class AttentionDecoder(BaseDecoder):
    def __init__(self, hidden_dim: int, node_out_dim: int, n_heads: int, aux_node: bool, bias: bool = False):
        """
        Attention decoder class.
        :param hidden_dim: int: The input dimension of the decoder.
        :param node_out_dim: int: The output dimension of the decoder.
        :param n_heads: int: The number of attention heads.
        :param bias: bool: Whether to use bias in the linear layers.
        """
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        assert hidden_dim % n_heads == 0, "Hidden dimension must be divisible by the number of heads."
        self.head_dim = node_out_dim*hidden_dim // n_heads

        self.node_out_dim = node_out_dim

        self.aux_node = aux_node

        self.W_q = nn.Linear(hidden_dim, node_out_dim*hidden_dim, bias=bias)
        self.W_kv = nn.Linear(hidden_dim, node_out_dim*2*hidden_dim, bias=bias)

        self.multi_head_combine = nn.Linear(node_out_dim*hidden_dim, node_out_dim*hidden_dim, bias=bias)

        # assert node_out_dim == 1, "Only single node output is supported for the AttentionDecoder."
        # Now we support multi-class outputs, multiplying the hidden dimension by the number of classes

    def forward(self, h):
        """
        Forward pass for the AttentionDecoder.
        :param h: torch.Tensor: The input embeddings. Shape: (batch_size, n_nodes, hidden_dim).
        """
        if self.aux_node:
            # Divide aux node from the node embeddings, and use it as the graph embedding (h_g)
            h, h_g = h[:, :-1, :], h[:, -1:, :]
        else:
            # Use the average of the node embeddings as the graph embedding (h_g)
            h_g = h.mean(dim=1)

        batch_size, n_nodes, _ = h.shape

        #  Multi-Head Attention. Q: aux_node, K: h, V: h
        q = self.W_q(h_g)
        k, v = self.W_kv(h).split(self.node_out_dim*self.hidden_dim, dim=2)
        q = q.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(batch_size, 1, self.node_out_dim*self.hidden_dim)

        mh_atten_out = self.multi_head_combine(y)
        # shape: (batch_size, 1, hidden_dim)

        mh_atten_out = mh_atten_out.view(batch_size, self.node_out_dim, self.hidden_dim)

        #  Single-Head Attention, for probability calculation
        y = torch.matmul(mh_atten_out, h.transpose(1, 2))
        # shape: (batch_size, 1, n_nodes)

        y = y.transpose(1, 2)
        # shape: (batch_size, n_nodes, 1)
        return y, h_g


# create a dictionary for the encoder and decoder classes
DECODER_DICT = {
    'linear': LinearDecoder,
    'attention': AttentionDecoder,
    'edge': LinearEdgeDecoder
}