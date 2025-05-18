import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import random
from typing import Tuple

import torch

from nco_lib.models.graph_transformer import EdgeInGTModel
from nco_lib.environment.env import State, Env, ConstructiveStoppingCriteria, ConstructiveReward
from nco_lib.environment.problem_def import ConstructiveProblem
from nco_lib.trainer.trainer import ConstructiveTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)



# %%
"""
Linear Ordering Problem (LOP) - Constructive method
The Linear Ordering Problem (LOP) is a combinatorial optimization problem where the goal is to find an optimal ordering of a set of elements.

We will define the LOP problem as a constructive problem, where in each step we will select an item to append to the ordering.
"""


# %%
# 1) Define the LOP constructive problem
class LOPConstructiveProblem(ConstructiveProblem):
    def _init_instances(self, state: State) -> State:
        """
        Here the user can define the generation of the instances for the LOP problem.
        These instances will be later used to define the node/edge features.
        The state class supports the adjacency matrix as a default data field.
        If any other data is needed, it should be stored in the state.data dictionary as done with the coordinates.
        The instances will generally have a shape of (batch_size, problem_size, problem_size, n) or (batch_size, problem_size, n), where n is the number of features.
        """
        if state.seed is not None:
            torch.manual_seed(state.seed)

        # Generate the graph weights randomly from a uniform distribution between 0 and 1
        weights = torch.rand(state.batch_size, state.problem_size, state.problem_size, device=state.device)

        # Zero the diagonal
        weights = weights * (1 - torch.eye(state.problem_size, device=state.device))

        state.data['weights'] = weights

        return state

    def _init_solutions(self, state: State) -> State:
        """
        Here the user can define the initialization of the solutions for the LOP problem.
        In improvement methods, the solutions are generally initialized as complete solutions.
        In the LOP it does not make sense to initialize multiple solutions per instance (with POMO).
        The solution will generally have a shape of (batch_size, pomo_size, problem_size).
        """
        # Set random seed if defined
        if state.seed is not None:
            torch.manual_seed(state.seed)

        # Initialize the solutions as empty
        state.solutions = torch.zeros((state.batch_size, state.pomo_size, 0), dtype=torch.long, device=state.device)

        return state

    def _init_features(self, state: State) -> State:
        """
        Here the user can define the initialization of the node features for the LOP problem.
        """
        state.node_features = torch.stack([torch.ones(state.batch_size, state.pomo_size, state.problem_size, device=state.device),
                                           torch.zeros(state.batch_size, state.pomo_size, state.problem_size, device=state.device)], dim=-1)

        weights = state.data['weights']
        up_tri = torch.triu(weights, 1) - torch.triu(weights.permute(0, 2, 1), 1)
        edges = up_tri - up_tri.permute(0, 2, 1)
        edges = edges.unsqueeze(1).repeat(1, state.pomo_size, 1, 1)
        state.edge_features = edges.unsqueeze(-1)

        return state

    def _init_mask(self, state: State) -> State:
        """
        Here the user can define the initialization of the mask.
        """

        state.mask = torch.zeros((state.batch_size, state.pomo_size, state.problem_size, 1), device=state.device)

        return state

    def _obj_function(self, state: State) -> torch.Tensor:
        """
        In this function, the user needs to define the objective function for the LOP problem.
        This function is called every improvement step.
        """

        obj_values = torch.zeros(state.batch_size, state.pomo_size, device=state.device)
        triu_indices = torch.triu_indices(state.problem_size, state.problem_size, offset=1)
        for b in range(state.batch_size):
            for p in range(state.pomo_size):
                mesh = torch.meshgrid([state.solutions[b, p], state.solutions[b, p]], indexing='ij')
                instance = state.data['weights'][b][mesh]
                obj_values[b, p] = instance[triu_indices[0, :], triu_indices[1, :]].sum(-1)

        return obj_values

    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to define how to update the node/edge features based on the new partial solutions.
        """
        # Initialize indices for batch and POMO dimensions
        batch_indices = torch.arange(state.batch_size, device=state.device)[:, None, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))
        pomo_indices = torch.arange(state.pomo_size, device=state.device)[None, :, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))

        # Only update the selected and last visited cities
        action = action[0]

        # Selected (0, 1) and non-selected (1, 0), modify from (1, 0) to (0, 1) for the selected cities
        node_features = state.node_features.clone()
        node_features[batch_indices, pomo_indices, action.unsqueeze(2), 0] = 0
        node_features[batch_indices, pomo_indices, action.unsqueeze(2), 1] = 1
        state.node_features = node_features
        return state

    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the solutions based on the selected actions.
        Actions are given in a tuple format, where the first part is the selected node and the second part is the selected class.
        In the case of the LOP, we only need the selected pair of nodes (edge); this is equivalent to having a single class.
        Therefore, only the first part of the action tuple is used.
        """
        # There is only one class (selected node) in the TSP, so only take the first part of the action tuple
        action = action[0]

        # Append the selected city to the solution
        state.solutions = torch.cat([state.solutions, action.unsqueeze(2)], dim=2)
        # state.solutions.shape: (batch_size, pomo_size, 0~problem_size)

        return state

    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the mask based on the selected actions (nodes).
        """
        action = action[0]

        # Mask the selected items
        batch_range = torch.arange(state.batch_size, device=state.device)[:, None].expand(state.batch_size, state.pomo_size)
        pomo_range = torch.arange(state.pomo_size, device=state.device)[None, :].expand(state.batch_size, state.pomo_size)
        state.mask[batch_range, pomo_range, action, :] = float('-inf')

        return state

    def _check_completeness(self, state: State) -> State:
        """
        This function is used to check if the solution is complete.
        """
        # Solution is complete if all items are selected
        state.is_complete = (state.solutions.size(2) == state.problem_size)
        return state


# %%
# 2) Define the environment, the model, and the trainer
lop_problem = LOPConstructiveProblem(device=device)

# Now, we define the environment for the LOP (permutation) using a constructive mode
lop_env = Env(problem=lop_problem,
              reward=ConstructiveReward(),
              stopping_criteria=ConstructiveStoppingCriteria(),
              device=device)

# Define the model based on 2 node features (2D coordinates) and
lop_model = EdgeInGTModel(decoder='linear', node_in_dim=2, edge_in_dim=1, aux_node=False,
                          logit_clipping=10.0).to(device)

# Define the RL training algorithm
lop_trainer = ConstructiveTrainer(model=lop_model,
                                  env=lop_env,
                                  optimizer=torch.optim.Adam(lop_model.parameters(), lr=1e-4),
                                  device=device)
# %%
# 3) Run training and inference for the Traveling Salesman Problem
train_results = lop_trainer.train(epochs=10, episodes=100, problem_size=20, batch_size=32, pomo_size=1,
                                  eval_problem_size=20, eval_batch_size=256, baseline_type='mean', verbose=True)

lop_trainer.inference(problem_size=20, batch_size=100, pomo_size=1, deterministic=True, seed=42, verbose=True)
