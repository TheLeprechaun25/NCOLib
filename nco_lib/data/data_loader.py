import torch


class DataLoader:
    def __init__(self, data_path: str, num_instances: int, random_shuffle: bool = False, device='cpu'):
        self.device = torch.device(device)
        self.data = None
        self.labels = None

        self.random_shuffle = random_shuffle

        self.data_path = data_path
        self.num_instances = num_instances

        self.cur_idx = 0

    def load_data(self,):
        raise NotImplementedError

    def generate_batch(self, batch_size):
        """
        Generate a batch of instances from loaded dataset.
        """
        if self.random_shuffle:
            random_indices = torch.randint(0, self.num_instances, batch_size)
            batch = self.data[random_indices]
        else:
            if self.cur_idx + batch_size > self.num_instances:
                # Overflow of index
                difference = self.cur_idx + batch_size - self.num_instances
                batch = self.data[self.cur_idx:] + self.data[:difference]
                self.cur_idx = difference
            else:
                # Normal operation
                batch = self.data[self.cur_idx:self.cur_idx+batch_size]
                self.cur_idx += batch_size

        return batch


def generate_random_graph(batch_size: int, problem_size: int, seed: int, edge_prob: float = 0.5,
                          device: str or torch.device = 'cpu') -> torch.Tensor:
    """
    Generate a batch of random graphs, represented as adjacency matrices. The graphs are generated by sampling
    random values from a uniform distribution and thresholding them at a given edge probability.

    :param batch_size: The number of graphs to generate. Type: int.
    :param problem_size: The number of nodes in each graph. Type: int.
    :param seed: The seed for reproducibility. Type: int or None.
    :param edge_prob: The probability of an edge existing between two nodes. Type: float.
    :param device: The device to use for computations. Type: str or torch.device.
    :return: torch.Tensor: A batch of adjacency matrices. Shape: (batch_size, problem_size, problem_size).
    """
    # Set the seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random values from a uniform distribution
    adj_mat = torch.rand(batch_size, problem_size, problem_size, device=device)

    # Threshold the values at the given edge probability
    adj_mat[adj_mat >= (1-edge_prob)] = 1
    adj_mat[adj_mat < (1-edge_prob)] = 0

    # make lower triangular part of the matrix zero
    adj_mat = adj_mat.tril(diagonal=-1)

    # Copy upper triangular part to lower triangular part
    adj_mat = adj_mat + adj_mat.permute(0, 2, 1)

    return adj_mat


def generate_tsp_graph(self, batch_size, problem_size, seed=None):
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
