import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import random
from typing import Tuple

import torch
import torch.nn.functional as F

from nco_lib.environment.actions import two_opt, bit_flip, insert, swap
from nco_lib.environment.env import State, Env, ImprovementReward, ImprovementStoppingCriteria
from nco_lib.environment.problem_def import ImprovementProblem
from nco_lib.models.graph_transformer import EdgeInOutGTModel
from nco_lib.trainer.trainer import ImprovementTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)



# %%
"""
Traveling Salesman Problem (TSP)
The Traveling Salesman Problem is a well-known combinatorial optimization problem that consists of finding the shortest
possible route that visits each city exactly once and returns to the origin city.

We will define the TSP problem as an improvement problem, where in each step we will select a pair of cities and relocate them.
In permutation problems, we can use several pairwise actions, such as 2-opt, swap, and insert.
"""


# %%
# 1) Define the TSP improvement problem
class TSPImprovementProblem(ImprovementProblem):
    def _init_instances(self, state: State) -> State:
        """
        Here the user can define the generation of the instances for the TSP problem.
        These instances will be later used to define the node/edge features.
        The state class supports the adjacency matrix as a default data field.
        If any other data is needed, it should be stored in the state.data dictionary as done with the coordinates.
        The instances will generally have a shape of (batch_size, problem_size, problem_size, n) or (batch_size, problem_size, n), where n is the number of features.
        """
        if state.seed is not None:
            torch.manual_seed(state.seed)

        # Generate the city coordinates as user-defined data
        state.data['coords'] = torch.rand(state.batch_size, state.problem_size, 2, device=state.device)

        # Also define the Euclidean distances to later be used as edge features
        state.data['distances'] = torch.cdist(state.data['coords'], state.data['coords'])

        return state

    def _init_solutions(self, state: State) -> State:
        """
        Here the user can define the initialization of the solutions for the TSP problem.
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
        In the improvement method for the TSP, we will mask the diagonal elements to avoid self-loops.
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
        In this function, the user needs to define the objective function for the TSP problem.
        This function is called every improvement step.
        """
        gathering_index = state.solutions.unsqueeze(3).expand(state.batch_size, state.pomo_size, state.problem_size, 2)
        # shape: (batch, pomo, problem, 2)

        seq_expanded = state.data['coords'][:, None, :, :].expand(state.batch_size, state.pomo_size, state.problem_size, 2)
        # shape: (batch, pomo, problem, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)

        return -travel_distances  # minimize the total distance  -> maximize the negative distance

    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to define how to update the node/edge features based on the new partial solutions.
        """
        # Initialize indices for batch and POMO dimensions
        batch_pomo_range = torch.arange(state.batch_size*state.pomo_size, device=state.device)

        # Use the 2D coordinates as node features
        state.node_features = state.data['coords'].unsqueeze(1).expand(state.batch_size, state.pomo_size, state.problem_size, 2)

        # Initialize edge solutions tensor
        edge_solutions = torch.zeros(state.batch_size*state.pomo_size, state.problem_size, state.problem_size,
                                     dtype=torch.float32, device=device)

        # Update edge solutions using advanced indexing
        solutions = state.solutions.view(-1, state.problem_size)  # shape: (batch_size*pomo_size, problem_size)
        solutions_plus_one = torch.cat([solutions[:, 1:], solutions[:, :1]], dim=1)
        edge_solutions[batch_pomo_range.unsqueeze(-1), solutions, solutions_plus_one] = 1

        # Make the edge solutions symmetric
        #edge_solutions = edge_solutions + edge_solutions.permute(0, 2, 1)

        # One-hot encoding of the edge solutions
        edge_solutions = F.one_hot(edge_solutions.long(), 2).float()

        # Reshape the edge solutions to have the POMO dimension
        edge_solutions = edge_solutions.view(state.batch_size, state.pomo_size, state.problem_size, state.problem_size, 2)

        # Use the distances as edge features
        distances = state.data['distances'].unsqueeze(1).expand(state.batch_size, state.pomo_size, state.problem_size, state.problem_size)
        state.edge_features = torch.cat([distances.unsqueeze(-1), edge_solutions], dim=-1)
        return state

    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the solutions based on the selected actions.
        Actions are given in a tuple format, where the first part is the selected node and the second part is the selected class.
        In the case of the TSP, we only need the selected pair of nodes (edge); this is equivalent to having a single class.
        Therefore, only the first part of the action tuple is used.
        """
        action = action[0]

        # Update the solutions using the 2-opt action
        state.solutions = two_opt(state.solutions, action)
        #state.solutions = insert(state.solutions, action)
        #state.solutions = swap(state.solutions, action)

        return state

    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the mask based on the selected actions (cities).
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
tsp_problem = TSPImprovementProblem(device=device)

# Now, we define the environment for the TSP (permutation)
tsp_env = Env(problem=tsp_problem,
              reward=ImprovementReward(positive_only=True, normalize=True),
              stopping_criteria=ImprovementStoppingCriteria(max_steps=200, patience=20),
              device=device)

# Define the model based on 2 node features (2D coordinates)
tsp_model = EdgeInOutGTModel(decoder='attention_edge', node_in_dim=2, edge_in_dim=3, edge_out_dim=1, aux_node=True,
                             logit_clipping=10.0).to(device)

# Define the RL training algorithm
tsp_trainer = ImprovementTrainer(model=tsp_model,
                                 env=tsp_env,
                                 optimizer=torch.optim.AdamW(tsp_model.parameters(), lr=1e-4),
                                 device=device)
# %%
# 3) Run training and inference for the Traveling Salesman Problem
ppo_args = {
    'ppo_epochs': 10,
    'ppo_clip': 0.2,
    'entropy_coef': 0.0,
    'ppo_update_batch_count': 4,
    'n_stored_states': 10,
}
train_results = tsp_trainer.train(epochs=100, episodes=5, problem_size=20, batch_size=32, pomo_size=4, learn_algo='ppo', eval_freq=5,
                                  eval_problem_size=20, eval_batch_size=256, baseline_type='pomo', ppo_args=ppo_args, verbose=True)

inference_results = tsp_trainer.inference(problem_size=20, batch_size=100, pomo_size=1, deterministic=True, seed=42, verbose=True)
