from abc import ABC, abstractmethod
from typing import Tuple

import torch


class State:
    def __init__(self, batch_size: int, problem_size: int or Tuple[int], pomo_size: int, node_features: torch.Tensor or None, adj_matrix: torch.Tensor or None,
                 edge_features: torch.Tensor or None, solutions: torch.Tensor or None, mask: torch.Tensor or None, is_complete: bool = False,
                 seed: int or None = None, device: str or torch.device = 'cpu', **kwargs):
        """
        The state of the environment. It contains all the information needed to represent the problem instance and the
        current solution.
        """
        self.data = kwargs  # User-defined attributes stored in a dictionary

        self.batch_size = batch_size  # Number of instances in the batch
        self.problem_size = problem_size  # Size of the problem (number of nodes)
        self.pomo_size = pomo_size  # Number of parallel initializations (Policy Optimization with Multiple Optima)

        self.adj_matrix = adj_matrix  # Adjacency matrix
        self.node_features = node_features  # Node features
        self.edge_features = edge_features   # Edge features

        self.solutions = solutions  # Current solutions
        self.mask = mask  # Mask to avoid selecting certain actions
        self.is_complete = is_complete  # Is the solution complete?

        self.memory_info = None  # Memory information
        self.last_action = None  # Last action taken

        self.seed = seed  # Seed for reproducibility
        self.device = device  # Device


class Problem(ABC):
    def __init__(self, device: str or torch.device):
        """
        Abstract class for the problem definition.
        :param device: str or torch.device: Device to use for computations.
        """
        self.device = torch.device(device)

        # Auxiliary state to get the dimensions of the node and edge features of the user-defined problem definition
        aux_state = State(batch_size=1, problem_size=9, pomo_size=1, node_features=None, adj_matrix=None, edge_features=None,
                          solutions=None, mask=None, is_complete=False, device=device)

        aux_state = self._init_instances(aux_state)
        aux_state = self._init_solutions(aux_state)
        aux_state = self._init_features(aux_state)
        self.node_in_dim = 0 if aux_state.node_features is None else aux_state.node_features.size(-1)
        self.edge_in_dim = 0 if aux_state.edge_features is None else aux_state.edge_features.size(-1)

    @abstractmethod
    def generate_state(self, batch_size: int, problem_size: int, pomo_size: int, seed: int or None = None) -> State:
        """
        Generate a batch of states for the problem. Follows the following steps:
        1 - Generate new graphs
        2 - Initialize the solution
        3 - Compute the initial objective value
        4 - Update state with features
        5 - Initialize mask

        :param batch_size: int: Number of instances in the batch.
        :param problem_size: int or list: Size of the problem (number of nodes).
        :param pomo_size: int: Number of parallel initializations (Policy Optimization with Multiple Optima).
        :param seed: int or None: Seed for reproducibility.
        :return: State: A state class with features representing the problem instance.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> (State, torch.Tensor):
        """
        Update the state with the given action. Follows the following steps:
        1 - Apply the action to the environment: update the solution
        2 or 3 - Compute the objective value
        2 or 3 - Check completeness
        4 - Update mask
        5 - Update features with new solution

        :param state: State: The current state of the environment.
        :param action: Tuple[torch.Tensor, torch.Tensor]: The action to apply to the environment. (selected node/edge, selected class)
        :return: (State, torch.Tensor): A state class with updated features and the objective value.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_instances(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_solutions(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_features(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_mask(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _obj_function(self, state: State) -> float:
        raise NotImplementedError

    @abstractmethod
    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _check_completeness(self, state: State) -> State:
        raise NotImplementedError


class ConstructiveProblem(Problem):
    def __init__(self, device: str or torch.device):
        """
        :param device: str or torch.device: Device to use for computations.
        """
        super().__init__(device)
        self.device = torch.device(device)

    def generate_state(self, problem_size: int, batch_size: int, pomo_size: int, seed: int or None = None) -> (State, torch.Tensor):
        """
        Generate a batch of states for the problem.

        Returns:
        - A state class with features representing the problem instance: node_features, edge_features, solution, mask...
        """

        # Initialize the state
        state = State(batch_size=batch_size, problem_size=problem_size, pomo_size=pomo_size, node_features=None,
                      adj_matrix=None, edge_features=None, solutions=None, mask=None, is_complete=False,
                      device=self.device, seed=seed)

        # 1 - Generate new graphs
        state = self._init_instances(state)

        # 2 - Initialize the solution
        state = self._init_solutions(state)

        # 3- Compute the initial objective value: in constructive frameworks, the objective value is typically computed at the end
        obj_value = torch.zeros(1, device=state.device)

        # 4 - Update state with features
        state = self._init_features(state)

        # 5 - Initialize mask
        state = self._init_mask(state)

        return state, obj_value

    def update_state(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> (State, torch.Tensor):
        """
        Update the state with the given action.

        Returns:
        - A state class with updated features.
        """

        # 1 - Apply the action to the environment: update the solution
        state = self._update_solutions(state, action)

        # 2 - Check completeness
        state = self._check_completeness(state)

        # 3 - Compute the objective value: only if the solution is complete
        if state.is_complete:
            obj_value = self._obj_function(state)
            # reshape to (batch_size*pomo_size)
            obj_value = obj_value.view(-1)
        else:
            obj_value = torch.empty(0, device=state.device)

        # 4 - Update mask
        state = self._update_mask(state, action)

        # 5 - Update features with new solutions
        state = self._update_features(state, action)

        return state, obj_value

    @abstractmethod
    def _init_instances(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_solutions(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_features(self, state: State) -> State:
        raise NotImplementedError

    def _init_mask(self, state: State) -> State:
        state.mask = torch.zeros((state.batch_size, state.problem_size, 1), device=state.device)
        return state

    @abstractmethod
    def _obj_function(self, state: State) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _check_completeness(self, state: State) -> State:
        raise NotImplementedError


class ImprovementProblem(Problem):
    def __init__(self, device: str or torch.device):
        """
        :param device: str or torch.device: Device to use for computations.
        """
        super().__init__(device)
        self.device = torch.device(device)

    def generate_state(self, problem_size: int, batch_size: int, pomo_size: int, seed: int = None):
        """
        Generate a batch of states for the problem.

        Returns:
        - A state class with features representing the problem instance: node_features, edge_features, solution, mask...
        """
        # Initialize the state
        state = State(batch_size=batch_size, problem_size=problem_size, pomo_size=pomo_size,
                      node_features=None, adj_matrix=None, edge_features=None,
                      solutions=None, mask=None, is_complete=False, device=self.device, seed=seed)

        # 1 - Generate new graphs
        state = self._init_instances(state)

        # 2 - Initialize the solution
        state = self._init_solutions(state)

        # 3- Compute the initial objective value
        obj_value = self._obj_function(state)

        # 4 - Update state with features
        state = self._init_features(state)

        # 5 - Initialize mask
        state = self._init_mask(state)

        return state, obj_value

    def update_state(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> (State, torch.Tensor):
        """
        Update the state with the given action.

        Returns:
        - A state class with updated features.
        """
        # 1 - Apply the action to the environment: update the solution
        state = self._update_solutions(state, action)

        # 2 - Compute the objective value: a solution is always complete in improvement frameworks
        obj_value = self._obj_function(state)

        # 3 - Update mask
        state = self._update_mask(state, action)

        # 5 - Update features with new solutions
        state = self._update_features(state, action)

        return state, obj_value

    @abstractmethod
    def _init_instances(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_solutions(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_features(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _init_mask(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _obj_function(self, state: State) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        raise NotImplementedError

    @abstractmethod
    def _check_completeness(self, state: State) -> State:
        raise NotImplementedError
