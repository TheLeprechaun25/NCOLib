import torch
from abc import ABC, abstractmethod
from nco_lib.environment.problem_def import Problem, State


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
    def __init__(self, normalize: bool = True):
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
    def __init__(self, positive_only: bool = True, normalize: bool = True):
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


class StoppingCriteria(ABC):
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


class Env(ABC):
    def __init__(self, problem: Problem, reward: Reward, stopping_criteria: StoppingCriteria, device: torch.device or str = 'cpu'):
        """
        Initialize the NCO environment.
        :param problem: The problem definition. Type: Problem.
        :param reward: The reward function. Type: Reward.
        :param stopping_criteria: The stopping criteria. Type: StoppingCriteria.
        :param device: The device to use for computations. Type: str.
        """
        self.problem = problem
        self.reward = reward
        self.stopping_criteria = stopping_criteria
        self.device = device

        self.state: State or None = None  # The current state of the environment
        self.iteration: int = 0  # The current iteration
        self.batch_range: torch.Tensor or None = None  # The range of batch sizes

    def reset(self, problem_size: int, batch_size: int, seed: int or None = None) -> (State, torch.Tensor, torch.Tensor, bool):
        """
        Reset the environment to an initial state and return it.
        :param problem_size: The size of the problem (number of nodes). Type: int.
        :param batch_size: The number of instances in the batch. Type: int.
        :param seed: The seed for reproducibility. Type: int or None.
        :return: The initial state, reward, and a boolean indicating if the episode has ended.
        """
        # Generate the initial state
        self.state, obj_value = self.problem.generate_state(problem_size=problem_size, batch_size=batch_size, seed=seed)

        # Calculate the reward
        reward = self.reward.reset(obj_value)

        # Reset the stopping criteria
        self.stopping_criteria.reset()

        return self.state, reward, obj_value, False

    def step(self, action: torch.Tensor) -> (State, torch.Tensor, torch.Tensor, bool):
        """
        Apply an action in the environment, modifying its state.
        :param action: The action to apply to the environment. Type: torch.Tensor.
        :return: The new state, reward, and a boolean indicating if the episode has ended.
        """
        # Update state with the action
        self.state, obj_value = self.problem.update_state(self.state, action)

        # Calculate the reward
        reward = self.reward.step(self.state, obj_value)

        # Check if the episode has ended
        done = self.stopping_criteria.step(self.state, obj_value)

        return self.state, reward, obj_value, done
