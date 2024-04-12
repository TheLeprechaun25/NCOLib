from abc import ABC, abstractmethod
import torch


class Generator(ABC):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)


class ErdosRenyiGraphGenerator(Generator):
    def __init__(self, p, device='cpu'):
        super(ErdosRenyiGraphGenerator, self).__init__(device)
        self.p = p

    def generate_unweighted(self, batch_size, problem_size, seed=None, device='cpu'):
        """
        Generate a batch of random graphs.
        """
        if seed is not None:
            torch.manual_seed(seed)

        adj_mat = torch.rand(batch_size, problem_size, problem_size, device=self.device)

        adj_mat[adj_mat > (1-self.p)] = 1
        adj_mat[adj_mat < (1-self.p)] = 0

        # make lower triangular part of the matrix zero
        adj_mat = adj_mat.triu(diagonal=-1)

        # Copy upper triangular part to lower triangular part
        adj_mat = adj_mat + adj_mat.permute(0, 2, 1)

        return adj_mat




class TSPGraphGenerator(Generator):
    def generate(self, batch_size, problem_size, seed=None):
        """
        Generate a batch of states for the TSP.

        Returns:
        - A state class with features representing the problem instance: node_features, edge_features, solution, mask...
        """

        if seed is not None:
            torch.manual_seed(seed)

        # Generate the city coordinates
        coords = torch.rand(batch_size, problem_size, 2, device=self.device)

        # Compute the Euclidean distance between cities
        distances = torch.cdist(coords, coords)

        return coords, distances
