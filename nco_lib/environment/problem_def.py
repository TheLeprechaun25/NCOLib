from abc import ABC, abstractmethod
from typing import Tuple
import torch


class State:
    def __init__(self, **kwargs):
        """
        Initialize the state of the environment.
        """
        self.data = kwargs  # User-defined attributes stored in a dictionary

        self.batch_size = None  # Number of instances in the batch
        self.problem_size = None  # Size of the problem (number of nodes)

        self.node_features = None  # Node features
        self.adj_matrix = None  # Adjacency matrix
        self.edge_features = None   # Edge features

        self.solutions = None   # Current solutions
        self.mask = None
        self.is_complete = None

        self.seed = None
        self.device = None


class Problem(ABC):
    def __init__(self, device: str or torch.device):
        #self.data_loader = data_loader
        self.device = torch.device(device)

    @abstractmethod
    def generate_state(self, batch_size: int, problem_size: int or list, seed=None) -> State:
        """
        Generate a batch of states for the problem.
        1 - Generate new graphs
        2 - Initialize the solution
        3 - Compute the initial objective value
        4 - Update state with features
        5 - Initialize mask

        Returns:
        - A state class with features representing the problem instance: node_features, edge_features, solution, mask...
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(self, state: State, action: torch.Tensor) -> Tuple[State, float]:
        """
        Update the state with the given action.
        1 - Apply the action to the environment: update the solution
        2/3 - Compute the objective value
        2/3 - Check completeness
        4 - Update mask
        5 - Update features with new solution
        Returns:
        - A state class with updated features.
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
    def _update_features(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_solutions(self, state: State, action: torch.Tensor) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_mask(self, state: State, action: torch.Tensor) -> State:
        raise NotImplementedError

    @abstractmethod
    def _check_completeness(self, state: State) -> State:
        raise NotImplementedError


class ConstructiveProblem(Problem):
    def generate_state(self, problem_size, batch_size, seed=None):
        """
        Generate a batch of states for the problem.

        Returns:
        - A state class with features representing the problem instance: node_features, edge_features, solution, mask...
        """
        state = State(batch_size=batch_size, problem_size=problem_size, device=self.device, seed=seed, is_complete=False)

        # 1 - Generate new graphs
        state = self._init_instances(state)

        # 2 - Initialize the solution
        state = self._init_solutions(state)

        # 3- Compute the initial objective value: in constructive frameworks, the objective value is typically computed at the end
        obj_value = 0

        # 4 - Update state with features
        state = self._init_features(state)

        # 5 - Initialize mask
        state = self._init_mask(state)

        return state, obj_value

    def update_state(self, state, action):
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
        else:
            obj_value = 0

        # 4 - Update mask
        state = self._update_mask(state, action)

        # 5 - Update features with new solution
        state = self._update_features(state)

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
    def _update_features(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_solutions(self, state: State, action: torch.Tensor) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_mask(self, state: State, action: torch.Tensor) -> State:
        raise NotImplementedError

    @abstractmethod
    def _check_completeness(self, state: State) -> State:
        raise NotImplementedError


class ImprovementProblem(Problem):
    def generate_state(self, problem_size, batch_size, seed=None):
        """
        Generate a batch of states for the problem.

        Returns:
        - A state class with features representing the problem instance: node_features, edge_features, solution, mask...
        """
        state = State(batch_size=batch_size, problem_size=problem_size, device=self.device, seed=seed, is_complete=True)

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

    def update_state(self, state, action):
        """
        Update the state with the given action.

        Returns:
        - A state class with updated features.
        """
        # 1 - Apply the action to the environment: update the solution
        state = self._update_solutions(state, action)

        # 2 - Compute the objective value: solution is always complete in improvement frameworks
        obj_value = self._obj_function(state)

        # 3 - Update mask
        state = self._update_mask(state, action)

        # 5 - Update features with new solution
        state = self._update_features(state)

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
    def _update_features(self, state: State) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_solutions(self, state: State, action: torch.Tensor) -> State:
        raise NotImplementedError

    @abstractmethod
    def _update_mask(self, state: State, action: torch.Tensor) -> State:
        raise NotImplementedError

    @abstractmethod
    def _check_completeness(self, state: State) -> State:
        raise NotImplementedError
