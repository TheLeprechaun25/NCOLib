from abc import ABC, abstractmethod
import torch


class DataLoader(ABC):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)

    def load_data(self, path):
        raise NotImplementedError

    def generate_batch(self, batch_size, problem_size, seed, **kwargs):
        raise NotImplementedError




def generate_random_graph(batch_size, problem_size, seed, edge_prob=0.5, device='cpu'):
    """
    Generate a batch of random graphs.
    """
    if seed is not None:
        torch.manual_seed(seed)

    adj_mat = torch.rand(batch_size, problem_size, problem_size, device=device)

    adj_mat[adj_mat > (1-edge_prob)] = 1
    adj_mat[adj_mat < (1-edge_prob)] = 0

    # make lower triangular part of the matrix zero
    adj_mat = adj_mat.triu(diagonal=-1)

    # Copy upper triangular part to lower triangular part
    adj_mat = adj_mat + adj_mat.permute(0, 2, 1)

    return adj_mat


