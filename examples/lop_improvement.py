import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import random
from typing import Tuple

import torch

from nco_lib.models.edge_gnn import EdgeGNNModel
from nco_lib.models.graph_transformer import GTModel, EdgeInGTModel, EdgeInOutGTModel

from nco_lib.environment.actions import two_opt, bit_flip, insert, swap
from nco_lib.environment.env import State, Env, ConstructiveStoppingCriteria, ConstructiveReward, ImprovementReward, ImprovementStoppingCriteria
from nco_lib.environment.problem_def import ConstructiveProblem, ImprovementProblem
from nco_lib.data.data_loader import generate_random_graph
from nco_lib.trainer.trainer import ConstructiveTrainer, ImprovementTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)



# %%
"""
Linear Ordering Problem (LOP) - Improvement method
The Linear Ordering Problem (LOP) is a combinatorial optimization problem where the goal is to find an optimal ordering of a set of elements.

We will define the LOP problem as an improvement problem, where in each step we will select a pair of nodes and relocate them.
In permutation problems, we can use several pairwise actions, such as 2-opt, swap, and insert.
"""


# %%
# 1) Define the LOP improvement problem
class LOPImprovementProblem(ImprovementProblem):
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
        The user can initialize multiple solutions per instance (with POMO).
        The solution will generally have a shape of (batch_size, pomo_size, problem_size).
        """
        # Set random seed if defined
        if state.seed is not None:
            torch.manual_seed(state.seed)

        # Initialize the solutions as random permutations
        state.solutions = torch.stack(
            [torch.randperm(state.problem_size, device=state.device) for _ in range(state.batch_size*state.pomo_size)])

        # Reshape the solutions to have the POMO dimension
        state.solutions = state.solutions.view(state.batch_size, state.pomo_size, state.problem_size)

        # Store the weight matrix ordered with the solution to later use it in the objective function and as features
        ordered_matrix = torch.zeros(state.batch_size, state.pomo_size, state.problem_size, state.problem_size, device=state.device)
        for b in range(state.batch_size):
            for p in range(state.pomo_size):
                mesh = torch.meshgrid([state.solutions[b, p], state.solutions[b, p]], indexing='ij')
                ordered_matrix[b, p] = state.data['weights'][b][mesh]

        state.data['ordered_weights'] = ordered_matrix
        return state

    def _init_features(self, state: State) -> State:
        """
        Here the user can define the initialization of the node features for the TSP problem.

        For the improvement method, the init_features and update_features will be the same, so we call it from here
        """
        action = (torch.empty(0), torch.empty(0))
        return self._update_features(state, action)

    def _init_mask(self, state: State) -> State:
        """
        Here the user can define the initialization of the mask.
        In the improvement method for the LOP, we will mask the diagonal elements to avoid self-loops.
        """

        # Mask the diagonal elements.
        mask = torch.zeros((state.batch_size, state.pomo_size, state.problem_size, state.problem_size, 1), device=state.device)
        row_indices = torch.arange(state.problem_size, device=state.device)
        mask[:, :, row_indices, row_indices, :] = -float('inf')
        # Reshape the mask to (batch_size, problem_size^2, 1)
        state.mask = mask.reshape(state.batch_size, state.pomo_size, -1, 1)

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
                instance = state.data['ordered_weights'][b, p]
                obj_values[b, p] = instance[triu_indices[0, :], triu_indices[1, :]].sum(-1)

        return obj_values

    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to define how to update the node/edge features based on the new partial solutions.
        """
        # Use a vector of ones as node features
        state.node_features = torch.ones(state.batch_size, state.pomo_size, state.problem_size, 1, device=state.device)

        # Initialize edge features
        state.edge_features = torch.zeros(state.batch_size, state.pomo_size, state.problem_size, state.problem_size, 2, device=state.device)
        for b in range(state.batch_size):
            for p in range(state.pomo_size):
                ordered_matrix = state.data['ordered_weights'][b, p]

                state.edge_features[b, p] = torch.stack([torch.triu(ordered_matrix, 1), torch.triu(ordered_matrix.permute(1, 0), 1)], dim=-1)

        return state

    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the solutions based on the selected actions.
        Actions are given in a tuple format, where the first part is the selected node and the second part is the selected class.
        In the case of the LOP, we only need the selected pair of nodes (edge); this is equivalent to having a single class.
        Therefore, only the first part of the action tuple is used.
        """
        action = action[0]

        # Update the solutions using the 2-opt action
        #state.solutions = two_opt(state.solutions, action)
        state.solutions = insert(state.solutions, action)
        #state.solutions = swap(state.solutions, action)

        # Store the weight matrix ordered with the solution to later use it in the objective function and as features
        ordered_matrix = torch.zeros(state.batch_size, state.pomo_size, state.problem_size, state.problem_size,
                                     device=state.device)
        for b in range(state.batch_size):
            for p in range(state.pomo_size):
                mesh = torch.meshgrid([state.solutions[b, p], state.solutions[b, p]], indexing='ij')
                instance = state.data['weights'][b][mesh]
                ordered_matrix[b, p] = instance

        state.data['ordered_weights'] = ordered_matrix

        return state

    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the mask based on the selected actions (nodes).
        """
        # The mask is static (only mask the diagonal elements), so no update is needed
        return state

    def _check_completeness(self, state: State) -> State:
        """
        This function is used to check if the solution is complete.
        """
        # In improvement problems, the solution is always complete
        state.is_complete = True
        return state


# %%
# 2) Define the environment, the model, and the trainer
lop_problem = LOPImprovementProblem(device=device)

# Now, we define the environment for the LOP (permutation) using a constructive mode
lop_env = Env(problem=lop_problem,
              reward=ImprovementReward(positive_only=False, normalize=True),
              stopping_criteria=ImprovementStoppingCriteria(max_steps=100, patience=5),
              device=device)

# Define the model based on 2 node features (2D coordinates) and
lop_model = EdgeGNNModel(decoder='edge', node_in_dim=1, edge_in_dim=2, edge_out_dim=1, aux_node=False,
                         logit_clipping=10.0).to(device)

# Define the RL training algorithm
lop_trainer = ImprovementTrainer(model=lop_model,
                                 env=lop_env,
                                 optimizer=torch.optim.Adam(lop_model.parameters(), lr=1e-4),
                                 device=device)
# %%
# 3) Run training and inference for the Traveling Salesman Problem
train_results = lop_trainer.train(epochs=10, episodes=100, problem_size=20, batch_size=32, pomo_size=3,
                                  eval_problem_size=20, eval_batch_size=256, baseline_type='mean', update_freq=10,
                                  verbose=True)

lop_trainer.inference(problem_size=20, batch_size=100, pomo_size=1, deterministic=True, seed=42, verbose=True)
