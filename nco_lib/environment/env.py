from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from nco_lib.environment.problem_def import Problem, State


class Reward(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, state, obj_value):
        raise NotImplementedError

    @abstractmethod
    def reset(self, obj_value):
        raise NotImplementedError


class ConstructiveReward(Reward):
    def __init__(self, normalize=True):
        super(ConstructiveReward, self).__init__()
        self.normalize = normalize

    def step(self, state, obj_value):
        if self.normalize:
            return obj_value/state.problem_size
        else:
            return obj_value

    def reset(self, obj_value):
        pass


class ImprovementReward(Reward):
    def __init__(self, positive_only=True, normalize=True):
        super(ImprovementReward, self).__init__()
        self.last_obj_value = None
        self.positive_only = positive_only
        self.normalize = normalize

    def step(self, state, obj_value):
        reward = obj_value - self.last_obj_value
        self.last_obj_value = obj_value
        if self.positive_only:
            reward = torch.clamp(reward, min=0)
        if self.normalize:
            return reward/state.problem_size
        else:
            return reward

    def reset(self, obj_value):
        self.last_obj_value = obj_value


class StoppingCriteria(ABC):
    def __init__(self):
        pass

    def step(self, state: State, obj_value: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class ConstructiveStoppingCriteria(StoppingCriteria):
    def step(self, state, obj_value):
        return state.is_complete

    def reset(self):
        pass


class ImprovementStoppingCriteria(StoppingCriteria):
    def __init__(self, max_steps, patience):
        super(ImprovementStoppingCriteria, self).__init__()
        self.max_steps = max_steps
        self.patience = patience
        self.cur_step = 0
        self.non_improving_steps = 0
        self.best_average_obj_value = float('-inf')

    def step(self, state, obj_value):
        avg_obj_value = obj_value.mean().item()
        self.cur_step += 1
        if avg_obj_value > self.best_average_obj_value:
            self.best_average_obj_value = avg_obj_value
            self.non_improving_steps = 0
        else:
            self.non_improving_steps += 1

        done = False
        if self.max_steps is not None:
            done = self.cur_step >= self.max_steps
        if self.patience is not None:
            done = done or self.non_improving_steps >= self.patience

        return done

    def reset(self):
        self.cur_step = 0
        self.non_improving_steps = 0
        self.best_average_obj_value = float('-inf')


class Env(ABC):
    def __init__(self, problem: Problem, reward: Reward, stopping_criteria: StoppingCriteria, device='cpu'):
        """
        Initialize the NCO environment.

        Parameters:
        - problem: The problem class.
        """
        self.problem = problem
        self.reward = reward
        self.stopping_criteria = stopping_criteria
        self.state = None

        self.iteration = 0
        self.batch_range = None
        self.device = device

    def reset(self, problem_size, batch_size, seed=None):
        """
        Reset the environment to an initial state and return it.
        Returns:
        - initial state: The initial state or observation of the environment.
        - initial reward: The reward of the initial state.
        - done: A boolean indicating if the episode has ended (False).
        """
        self.state, obj_value = self.problem.generate_state(problem_size=problem_size, batch_size=batch_size, seed=seed)
        reward = self.reward.reset(obj_value)
        self.stopping_criteria.reset()
        return self.state, reward, obj_value, False

    def step(self, action):
        """
        Apply an action in the environment, modifying its state.

        Parameters:
        - action: The action to be applied.

        Returns:
        - observation: The new state of the environment after applying the action.
        - reward: The reward resulting from the action.
        - done: A boolean indicating if the episode has ended.
        """
        self.state, obj_value = self.problem.update_state(self.state, action)
        reward = self.reward.step(self.state, obj_value)
        done = self.stopping_criteria.step(self.state, obj_value)
        return self.state, reward, obj_value, done

'''
class ConstructiveEnv(Env, ABC):
    def __init__(self, generate_graph_fn, generate_features_fn, obj_function, mask_fn, device='cpu'):
        super().__init__(generate_graph_fn, generate_features_fn, obj_function, mask_fn, device)
        self.selected_mask = None

    def reset(self, problem_size, batch_size, seed=None):
        """
        Reset the environment to an initial state and return it.
        Returns:
        - initial state: The initial state or observation of the environment.
        - initial reward: The reward of the initial state (0).
        - done: A boolean indicating if the episode has ended (False).
        """
        # Update the environment state
        self.state.problem_size = problem_size
        self.state.batch_size = batch_size
        self.state.seed = seed

        # 1 - Generate new graphs
        self.state = self.problem.generate_graph_fn(self.state)

        # 2 - Initialize the solutions
        self.problem.initialize_solution()

        # 3 - Update state with features
        self.state = self.problem.generate_features_fn(self.state)

        # 4 - Initialize mask
        self._initialize_mask()

        self.iteration = 0
        self.batch_range = torch.arange(batch_size, device=self.device)
        return self.state, 0, False

    def step(self, action):
        """
        Apply an action to the environment, modifying its state.

        Parameters:
        - action: The action to be applied.

        Returns:
        - observation: The new state of the environment after applying the action.
        - reward: The reward resulting from the action.
        - done: A boolean indicating if the episode has ended.
        """
        self.iteration += 1

        # Apply the action to the environment: update the solution and mask
        self._update_solution_and_mask(action)

        # Update features with new solution
        self.state = self.generate_features_fn(self.state)

        self.state.done = self._check_done()  # first check if the episode has ended, then the reward
        if self.state.done:
            obj_value = self.obj_function(self.state)
        else:
            obj_value = 0
        return self.state, obj_value, self.state.done

    @abstractmethod
    def _initialize_solution(self):
        """
        Initialize a solution for the problem. To be implemented by subclasses.

        Returns:
        - A valid initial solution.
        """
        pass

    @abstractmethod
    def _initialize_mask(self):
        """
        Initialize a mask for the problem. To be implemented by subclasses.

        Returns:
        - A mask for the initial state.
        """
        pass

    @abstractmethod
    def _update_solution_and_mask(self, action):
        """
        Apply an action to modify the current solution. To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _check_done(self):
        """
        Check if the current episode has ended. To be implemented by subclasses.

        Returns:
        - True if the episode has ended, False otherwise.
        """
        pass


class ImprovementEnv(Env, ABC):
    def __init__(self, generate_graph_fn, generate_features_fn, obj_function, mask_fn, ActionClass: Action, init_fn=None, max_steps=10, patience=5, device='cpu'):
        super().__init__(generate_graph_fn, generate_features_fn, obj_function, mask_fn, device)
        self.init_fn = init_fn
        self.Action = ActionClass
        self.stopping_criteria = StoppingCriteria(max_steps, patience)

    def reset(self, problem_size, batch_size, seed=None):
        """
        Reset the environment to an initial state and return it.
        Returns:
        - initial state: The initial state or observation of the environment.
        - initial reward: The reward of the initial state (0).
        - done: A boolean indicating if the episode has ended (False).
        """
        # Update the environment state
        self.state.problem_size = problem_size
        self.state.batch_size = batch_size
        self.state.seed = seed

        # 1 - Generate new graphs
        self.state = self.generate_graph_fn(self.state)

        # 2 - Initialize the solution
        self._initialize_solution()

        # 3- Compute the initial objective value
        obj_value = self.obj_function(self.state)

        # 4 - Update state with features
        self.state = self.generate_features_fn(self.state)

        # 5 - Initialize mask
        self._initialize_mask()

        # Reset stopping criteria
        self.stopping_criteria.reset()

        self.iteration = 0
        self.batch_range = torch.arange(batch_size, device=self.device)
        return self.state, obj_value, False

    def step(self, action):
        """
        Apply an action to the environment, modifying its state.

        Parameters:
        - action: The action to be applied.

        Returns:
        - observation: The new state of the environment after applying the action.
        - reward: The reward resulting from the action.
        - done: A boolean indicating if the episode has ended.
        """
        self.iteration += 1

        # Apply the action to the environment: update the solution and mask
        self._update_solution_and_mask(action)

        # Update features with new solution
        self.state = self.generate_features_fn(self.state)

        # Evaluate the objective value of the new solution
        obj_value = self.obj_function(self.state)

        # Check if the episode has ended
        self.state.done = self._check_done(obj_value)  # first check if the episode has ended, then the reward
        return self.state, obj_value, self.state.done

    @abstractmethod
    def _initialize_solution(self):
        """
        Initialize a solution for the problem. To be implemented by subclasses.

        Returns:
        - A valid initial solution.
        """
        pass

    @abstractmethod
    def _initialize_mask(self):
        """
        Initialize a mask for the problem. To be implemented by subclasses.

        Returns:
        - A mask for the initial state.
        """
        if self.mask_fn is not None:
            self.state.mask = self.mask_fn(self.state)
        else:
            self.state.mask = torch.zeros((self.state.batch_size, self.state.problem_size, 1), device=self.device)

    def _update_solution_and_mask(self, action):
        """
        Apply an action to modify the current solution. To be implemented by subclasses.
        """
        self.state.solutions = self.Action(self.state.solutions, action, self.batch_range)

        if self.mask_fn is not None:
            self.state.mask = self.mask_fn(self.state)

    def _check_done(self, reward):
        """
        Check if the current episode has ended. To be implemented by subclasses.

        Returns:
        - True if the episode has ended, False otherwise.
        """
        return self.stopping_criteria(reward)


class NARHeatmapEnv(Env):
    def __init__(self, generate_graph_fn, generate_features_fn, obj_function, mask_fn, device='cpu'):
        super().__init__(generate_graph_fn, generate_features_fn, obj_function, mask_fn, device)

    def reset(self, problem_size, batch_size, seed=None):
        """
        Reset the environment to an initial state and return it.
        Returns:
        - initial state: The initial state or observation of the environment.
        - initial reward: The reward of the initial state (0).
        - done: A boolean indicating if the episode has ended (False).
        """
        # Update the environment state
        self.state.problem_size = problem_size
        self.state.batch_size = batch_size
        self.state.seed = seed

        # 1 - Generate new graphs
        self.state = self.generate_graph_fn(self.state)

        # 2 - Update state with features
        self.state = self.generate_features_fn(self.state)

        self.batch_range = torch.arange(batch_size, device=self.device)
        return self.state, 0, False

    def step(self, logits, deterministic=True):
        """
        Apply an action to the environment, modifying its state.

        Parameters:
        - logits: The logits of the model.
        - deterministic: A boolean indicating if the selection of variable should be deterministic.

        Returns:
        - observation: The new state of the environment after applying the action.
        - reward: The reward resulting from the action.
        - done: A boolean indicating if the episode has ended.
        """
        if deterministic:
            self.greedy_search(logits)
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)


        obj_value = self.obj_function(self.state)

        return self.state, obj_value, True

    def greedy_search(self, logits):
        pass


class ConstructivePermutationEnv(ConstructiveEnv):
    def _initialize_solution(self):
        """
        Initialize a solution for the problem.

        Returns:
        - A valid initial solution.
        """
        self.state.solutions = torch.zeros((self.state.batch_size, 0), device=self.device)

    def _initialize_mask(self):
        """
        Initialize a mask for the problem. To be implemented by subclasses.

        Returns:
        - A mask for the initial state.
        """
        self.selected_mask = torch.zeros((self.state.batch_size, self.state.problem_size, 1), device=self.device)

        if self.mask_fn is not None:
            self.state.mask = self.selected_mask + self.mask_fn(self.state)
        else:
            self.state.mask = self.selected_mask

    def _update_solution_and_mask(self, action):
        """
        Apply an action to modify the current solution. To be implemented by subclasses.
        """
        self.state.solutions = torch.cat((self.state.solutions, action[:, None]), dim=1)

        self.selected_mask[self.batch_range, action, :] = float('-inf')

        if self.mask_fn is not None:
            self.state.mask = self.selected_mask + self.mask_fn(self.state)
        else:
            self.state.mask = self.selected_mask

    def _check_done(self):
        """
        Check if the current episode has ended. To be implemented by subclasses.

        Returns:
        - True if the episode has ended, False otherwise.
        """
        return self.state.solutions.size(1) == self.state.node_features.size(1)


class ConstructiveNClassEnv(ConstructiveEnv):
    def __init__(self, generate_graph_fn, generate_features_fn, obj_function, mask_fn, n_classes, device='cpu'):
        super().__init__(generate_graph_fn, generate_features_fn, obj_function, mask_fn, device)
        self.state.n_classes = n_classes

    def _initialize_solution(self):
        """
        Initialize a solution for the problem. To be implemented by subclasses.

        Returns:
        - A valid initial solution.
        """
        # Solutions are initialized as a tensor of zeros (unassigned nodes). Each step will assign a node to a class.
        self.state.solutions = torch.zeros((self.state.batch_size, self.state.problem_size), device=self.device)

    def _initialize_mask(self):
        """
        Initialize a mask for the problem. To be implemented by subclasses.

        Returns:
        - A mask for the initial state.
        """
        self.selected_mask = torch.zeros((self.state.batch_size, self.state.problem_size, self.state.n_classes), device=self.device)

        if self.mask_fn is not None:
            self.state.mask = self.selected_mask + self.mask_fn(self.state)
        else:
            self.state.mask = self.selected_mask

    def _update_solution_and_mask(self, action):
        """
        Apply an action to modify the current solution.
        """
        # Action is the node-to-class assignment [batch_size, problem_size*n_classes]
        classes = action % self.state.n_classes
        nodes = action // self.state.n_classes
        # Update solutions based on actions
        self.state.solutions[self.batch_range, nodes] = classes.float() + 1

        # Update node features with new solutions
        self.state.node_features = F.one_hot(self.state.solutions.long(), self.state.n_classes+1).float()

        self.selected_mask[self.batch_range, nodes, :] = float('-inf')

        if self.mask_fn is not None:
            self.state.mask = self.selected_mask + self.mask_fn(self.state)
        else:
            self.state.mask = self.selected_mask

    def _check_done(self):
        """
        Check if the current episode has ended. To be implemented by subclasses.

        Returns:
        - True if the episode has ended, False otherwise.
        """
        return (self.state.solutions == 0).sum() == 0


class ImprovementPermutationEnv(ImprovementEnv):
    def _initialize_solution(self):
        """
        Initialize a solution for the problem. To be implemented by subclasses.

        Returns:
        - A valid initial solution (permutation).
        """
        if self.init_fn is not None:
            self.state.solutions = self.init_fn(self.state)
        else:
            if self.state.seed is not None:
                torch.manual_seed(self.state.seed)

            self.state.solutions = torch.stack([torch.randperm(self.state.problem_size, device=self.device) for _ in range(self.state.batch_size)])

    def _initialize_mask(self):
        """
        Initialize a mask for the problem. To be implemented by subclasses.

        Returns:
        - A mask for the initial state.
        """
        if self.mask_fn is not None:
            self.state.mask = self.mask_fn(self.state)
        else:
            self.state.mask = torch.zeros((self.state.batch_size, self.state.problem_size**2, 1), device=self.device)


class ImprovementBinaryEnv(ImprovementEnv):
    def _initialize_solution(self):
        """
        Initialize a solution for the problem. To be implemented by subclasses.

        Returns:
        - A valid initial solution.
        """
        if self.init_fn is not None:
            self.state.solutions = self.init_fn(self.state)
        else:
            if self.state.seed is not None:
                torch.manual_seed(self.state.seed)
            self.state.solutions = torch.randint(0, 2, (self.state.batch_size, self.state.problem_size), device=self.device)

    def _initialize_mask(self):
        """
        Initialize a mask for the problem. To be implemented by subclasses.

        Returns:
        - A mask for the initial state.
        """
        if self.mask_fn is not None:
            self.state.mask = self.mask_fn(self.state)
        else:
            self.state.mask = torch.zeros((self.state.batch_size, self.state.problem_size, 1), device=self.device)

'''
