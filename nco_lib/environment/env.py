import torch
from typing import Tuple
from copy import deepcopy
from abc import ABC, abstractmethod

from nco_lib.environment.problem_def import Problem, State
from nco_lib.environment.memory import Memory, NoMemory, MarcoMemory, LastActionMemory
from nco_lib.data.data_loader import DataLoader


class Reward(ABC):
    def __init__(self):
        """
        Reward parent class with two abstract methods:
            step: for calculating the reward at each step.
            reset: for resetting the reward.
        """
        pass

    @abstractmethod
    def step(self, state: State, obj_value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reset(self, obj_value: torch.Tensor):
        raise NotImplementedError


class ConstructiveReward(Reward):
    def __init__(self, normalize: bool = False):
        """
        Constructive reward function. The reward is the objective value of the current state.
        :param normalize: If True, the reward is normalized by the problem size. Type: bool.
        """
        super(ConstructiveReward, self).__init__()
        self.normalize = normalize # Normalize the reward by the problem size

    def step(self, state: State, obj_value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the reward at each step.
        :param state: The current state of the environment. Type: State.
        :param obj_value: The objective value of the current state. Type: torch.Tensor.
        :return: The reward of the current state. Type: torch.Tensor.
        """

        if self.normalize:
            return obj_value/state.problem_size
        else:
            return obj_value

    def reset(self, obj_value: torch.Tensor):
        """
        Reset the reward.
        """
        pass


class ImprovementReward(Reward):
    def __init__(self, positive_only: bool = True, normalize: bool = False):
        """
        Improvement reward function. The reward is the difference between the current objective value and the previous one.
        :param positive_only: If True, the reward is clamped to zero if it is negative. Type: bool.
        :param normalize: If True, the reward is normalized by the problem size. Type: bool.
        """
        super(ImprovementReward, self).__init__()
        self.positive_only = positive_only
        self.normalize = normalize
        self.last_obj_value = None  # The objective value of the previous state

    def step(self, state, obj_value):
        """
        Calculate the reward at each step.
        :param state: The current state of the environment. Type: State.
        :param obj_value: The objective value of the current state. Type: torch.Tensor.
        :return: The reward of the current state. Type: torch.Tensor.
        """
        reward = obj_value - self.last_obj_value
        self.last_obj_value = obj_value
        if self.positive_only:
            reward = torch.clamp(reward, min=0)
        if self.normalize:
            return reward/state.problem_size
        else:
            return reward

    def reset(self, obj_value: torch.Tensor):
        """
        Reset the reward. Initialize the objective value of the previous state.
        :param obj_value: The objective value of the initial state. Type: torch.Tensor.
        """
        self.last_obj_value = obj_value


class StoppingCriteria:
    def __init__(self):
        """
        Stopping Criteria parent class with two abstract methods:
            step: for checking if the episode has ended.
            reset: for resetting the stopping criteria.
        """
        pass

    def step(self, state: State, obj_value: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class ConstructiveStoppingCriteria(StoppingCriteria):
    def __init__(self):
        """
        Initialize the stopping criteria for constructive methods.
        """
        super(ConstructiveStoppingCriteria, self).__init__()

    def step(self, state: State, obj_value: torch.Tensor) -> bool:
        """
        Check if the episode has ended. The episode ends when the solution is complete.
        :param state: The current state of the environment. Type: State.
        :param obj_value: The objective value of the current state. Type: torch.Tensor.
        """
        return state.is_complete

    def reset(self):
        pass


class ImprovementStoppingCriteria(StoppingCriteria):
    def __init__(self, max_steps: int, patience: int):
        """
        Initialize the stopping criteria for improvement methods.
        :param max_steps: The maximum number of improvement steps. Type: int.
        :param patience: The number of non-improving steps before the episode ends. Type: int.
        """
        super(ImprovementStoppingCriteria, self).__init__()
        self.max_steps = max_steps
        self.patience = patience

        self.cur_step = 0  # Current step
        self.non_improving_steps = 0  # Number of non-improving steps
        self.best_average_obj_value = float('-inf')  # Best average objective value so far

    def step(self, state: State, obj_value: torch.Tensor) -> bool:
        """
        Check if the episode has ended. The episode ends when the maximum number of steps is reached or the patience is exceeded.
        :param state: The current state of the environment. Type: State.
        :param obj_value: The objective value of the current state. Type: torch.Tensor.
        :return: A boolean indicating if the episode has ended.
        """
        # Increment the current step
        self.cur_step += 1

        # Calculate the average objective value
        avg_obj_value = obj_value.mean().item()

        # Check if the average objective value has improved
        if avg_obj_value > self.best_average_obj_value:
            self.best_average_obj_value = avg_obj_value
            self.non_improving_steps = 0
        else:
            self.non_improving_steps += 1

        # Check if the episode has ended
        done = False
        # Check if the maximum number of steps is reached
        if self.max_steps is not None:
            done = self.cur_step >= self.max_steps
        # Check if the patience is exceeded
        if self.patience is not None:
            done = done or self.non_improving_steps >= self.patience

        return done

    def reset(self):
        self.cur_step = 0
        self.non_improving_steps = 0
        self.best_average_obj_value = float('-inf')


class Env:
    def __init__(self, problem: Problem, reward: Reward, stopping_criteria: StoppingCriteria,
                 memory: Memory = None, data_loader: DataLoader or None = None, device: torch.device or str = 'cpu'):
        """
        Initialize the NCO environment.
        :param problem: The problem definition. Type: Problem.
        :param reward: The reward function. Type: Reward.
        :param stopping_criteria: The stopping criteria. Type: StoppingCriteria.
        :param memory: The memory to use for the environment. Type: Memory.
        :param data_loader: The dataset loader class. Type: DataLoader.
        :param device: The device to use for computations. Type: str.
        """
        self.problem = problem
        self.reward = reward
        self.stopping_criteria = stopping_criteria
        self.memory = memory
        self.device = device

        self.state: State or None = None  # The current state of the environment

        self.data_loader: DataLoader or None = data_loader  # The dataset loader used to train.
        if data_loader is not None:
            self.data_loader.load_data()  # Load the dataset

        self.iteration: int = 0  # The current iteration

        if memory is None or isinstance(memory, NoMemory):
            self.memory = NoMemory() # If no memory is provided, use NoMemory
            self.mem_dim = 0
        elif isinstance(memory, LastActionMemory):
            self.mem_dim = 1
        elif isinstance(memory, MarcoMemory):
            self.mem_dim = 2
        else:
            raise ValueError("Memory type not recognized: {}".format(memory))

    def set(self, state: State, obj_value: torch.Tensor):
        """
        Set the environment to a specific state.
        :param state: The state of the environment. Type: State.
        :param obj_value: The objective value of the state. Type: torch.Tensor.
        """
        self.state = deepcopy(state)

        reward = self.reward.reset(obj_value)

        # Reset the stopping criteria
        self.stopping_criteria.reset()

        return self.state, reward, obj_value, False


    def reset(self, problem_size: int or range, batch_size: int, pomo_size: int, seed: int or None = None) -> (State, torch.Tensor, torch.Tensor, bool):
        """
        Reset the environment to an initial state and return it.
        :param problem_size: The size of the problem (number of nodes) or a range of problem sizes. Type: int or range.
        :param batch_size: The number of instances in the batch. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param seed: The seed for reproducibility. Type: int or None.
        :return: The initial state, reward, and a boolean indicating if the episode has ended.
        """
        # Set the current problem size
        assert isinstance(problem_size, int) or isinstance(problem_size, range), "Problem size must be an integer or a range."
        cur_problem_size = problem_size if isinstance(problem_size, int) else torch.randint(low=problem_size.start, high=problem_size.stop, size=(1,)).item()

        # Generate the initial state
        self.state, obj_value = self.problem.generate_state(problem_size=cur_problem_size, batch_size=batch_size,
                                                            pomo_size=pomo_size, seed=seed)

        # add the memory information to the state
        if isinstance(self.memory, NoMemory):
            self.state.memory_info = None
        else:
            # add a dummy tensor to keep the dimensions
            self.state.memory_info = torch.zeros((batch_size, pomo_size, cur_problem_size, self.mem_dim), device=self.device)

        # Check whether the DataLoader will be used
        if self.data_loader is not None:
            # Generate a batch from the dataset
            self.state.data['loaded_data'] = self.data_loader.generate_batch(batch_size=batch_size)

        # Calculate the reward
        reward = self.reward.reset(obj_value)

        # Empty the memory
        self.memory.clear_memory()

        # Reset the stopping criteria
        self.stopping_criteria.reset()

        return self.state, reward, obj_value, False

    def step(self, action: Tuple[torch.Tensor, torch.Tensor]) -> (State, torch.Tensor, torch.Tensor, bool):
        """
        Apply an action in the environment, modifying its state.
        :param action: The action to apply to the environment in a tuple (selected node/edge, selected class). Type: Tuple[torch.Tensor, torch.Tensor].
        :return: The new state, reward, and a boolean indicating if the episode has ended.
        """
        # Store the previous state and the performed action in memory
        self.memory.save_in_memory(state=self.state, action=action)

        # Update state with the action
        self.state, obj_value = self.problem.update_state(self.state, action)

        # Gather the information from memory. K-nearest neighbors of current solutions
        self.state.memory_info, revisited, avg_sim, max_sim = self.memory.get_knn(self.state, k=self.memory.k)

        # Calculate the reward
        reward = self.reward.step(self.state, obj_value)

        # Update the reward with revisiting punishments if revisited is not None (only when using memory)
        if revisited is not None:
            revisited_idx = revisited != 0
            reward[revisited_idx] -= self.memory.repeat_punishment * revisited[revisited_idx]

        # Check if the episode has ended
        done = self.stopping_criteria.step(self.state, obj_value)

        # Store the action in state
        self.state.last_action = action

        return self.state, reward, obj_value, done
