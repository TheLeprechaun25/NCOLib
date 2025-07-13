import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import random
from typing import Tuple

import torch
from nco_lib.environment.env import State, Env, ConstructiveStoppingCriteria, ConstructiveReward
from nco_lib.environment.problem_def import ConstructiveProblem
from nco_lib.models.graph_transformer import GTModel
from nco_lib.models.gcn import GCNModel
from nco_lib.trainer.trainer import ConstructiveTrainer


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

We will define the TSP problem as a constructive problem, where in each step we will select a city to visit.
"""


# %%
# 1) Define the TSP constructive problem
class TSPConstructiveProblem(ConstructiveProblem):
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
        coords = torch.rand(state.batch_size, state.problem_size, 2, device=state.device)
        state.data['coords'] = coords

        # Fully connected or kNN sparse graph
        state.adj_matrix = torch.ones(state.batch_size, state.problem_size, state.problem_size, device=state.device)

        """# compute all pairwise distances: [B, N, N] `torch.cdist` gives sqrt-sum-squared in last dim
        dists = torch.cdist(coords, coords, p=2)  # Euclidean

        # For each node i, pick the k+1 smallest distances (including self at 0)
        #    topk returns the k+1 smallest (so we can drop self):
        k = 4
        knn = dists.topk(k=k + 1, largest=False)  # values, indices
        knn_idx = knn.indices  # [B, N, k+1]

        # 4) drop the first column (self-loop at distance 0)
        knn_idx = knn_idx[..., 1:]  # now [B, N, k]

        # 5) build empty adjacency and scatter 1’s at (i, neighbors)
        adj = torch.zeros(state.batch_size, state.problem_size, state.problem_size, device=state.device)
        batch_idx = torch.arange(state.batch_size, device=state.device)[:, None, None]  # [B,1,1]
        node_idx = torch.arange(state.problem_size, device=state.device)[None, :, None]  # [1,N,1]

        # set adj[b,i, knn_idx[b,i,j]] = 1
        adj[batch_idx, node_idx, knn_idx] = 1

        # 6) (optional) make the graph undirected by symmetrizing
        state.adj_matrix = (adj + adj.transpose(1, 2) > 0).float()"""

        return state

    def _init_solutions(self, state: State) -> State:
        """
        Here the user can define the initialization of the solutions for the TSP problem.
        In constructive methods, the solutions are generally initialized empty.
        However, for the TSP, we can select a random city as the starting point.
        In fact, the user can initialize multiple constructions per instance (with POMO).
        The solution will generally have a shape of (batch_size, pomo_size, 0~problem_size).
        """

        # We will set random city initializations from 0 to problem_size-1 (problem size is the number of cities)
        state.solutions = torch.randint(0, state.problem_size, (state.batch_size, state.pomo_size, 1), device=state.device)

        return state

    def _init_features(self, state: State) -> State:
        """
        Here the user can define the initialization of the node features for the TSP problem.

        For the TSP, we will use the coordinates, whether the city is selected or not, and whether it is the first or last city in a one-hot encoding.
        In this case, the node features will have a shape of (batch_size, pomo_size, problem_size, n), where n is the number of features.
        """
        # Initialize indices for batch and POMO dimensions
        batch_range = torch.arange(state.batch_size, device=state.device)[:, None].expand(state.batch_size, state.pomo_size)
        pomo_range = torch.arange(state.pomo_size, device=state.device)[None, :].expand(state.batch_size, state.pomo_size)
        batch_indices = torch.arange(state.batch_size, device=state.device)[:, None, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))
        pomo_indices = torch.arange(state.pomo_size, device=state.device)[None, :, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))

        # Create the coordinates for each city, expanded for the POMO dimension
        pomo_coords = state.data['coords'].unsqueeze(1).expand(state.batch_size, state.pomo_size, state.problem_size, 2)

        # One hot encoding for the selected cities: selected (0, 1) and non-selected (1, 0)
        selected = torch.zeros(state.batch_size, state.pomo_size, state.problem_size, 2, device=device)
        selected[batch_range, pomo_range, :, 0] = 1
        selected[batch_indices, pomo_indices, state.solutions.long(), 0] = 0
        selected[batch_indices, pomo_indices, state.solutions.long(), 1] = 1

        # One hot encoding for the first cities
        first_selected = torch.zeros(state.batch_size, state.pomo_size, state.problem_size, 1, device=device)
        first_selected[batch_range, pomo_range, state.solutions[:, :, 0].long(), 0] = 1

        # One hot encoding for the last cities
        last_selected = torch.zeros(state.batch_size, state.pomo_size, state.problem_size, 1, device=device)
        last_selected[batch_range, pomo_range, state.solutions[:, :, -1].long(), 0] = 1

        # Concatenate the features
        state.node_features = torch.cat([pomo_coords, selected, first_selected, last_selected], dim=-1)
        return state

    def _init_mask(self, state: State) -> State:
        """
        Here the user can define the initialization of the mask.
        In the TSP, the mask will be used to prevent the model from selecting the same city multiple times.
        Therefore, we need to mask the selected cities for each construction.
        """

        # Use POMO: Mask the selected cities for each construction
        batch_range = torch.arange(state.batch_size, device=state.device)[:, None].expand(state.batch_size, state.pomo_size)
        pomo_range = torch.arange(state.pomo_size, device=state.device)[None, :].expand(state.batch_size, state.pomo_size)

        # Get the selected cities from the POMO solutions
        action = state.solutions.squeeze(2)

        # Initialize the mask to zeros
        state.mask = torch.zeros((state.batch_size, state.pomo_size, state.problem_size, 1), device=state.device)

        # Mask the selected cities
        state.mask[batch_range, pomo_range, action, :] = float('-inf')

        return state

    def _obj_function(self, state: State) -> torch.Tensor:
        """
        In this function, the user needs to define the objective function for the TSP problem.
        This function is called only once the solution is completed.
        """

        gathering_index = state.solutions.unsqueeze(3).expand(state.batch_size, -1, state.problem_size, 2)
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
        batch_indices = torch.arange(state.batch_size, device=state.device)[:, None, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))
        pomo_indices = torch.arange(state.pomo_size, device=state.device)[None, :, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))

        # Only update the selected and last visited cities
        action = action[0]

        # Selected (0, 1) and non-selected (1, 0), modify from (1, 0) to (0, 1) for the selected cities
        node_features = state.node_features.clone()
        node_features[batch_indices, pomo_indices, action.unsqueeze(2), 2] = 0

        node_features[batch_indices, pomo_indices, action.unsqueeze(2), 3] = 1

        # Update the node features
        state.node_features = node_features

        # Last visited feature to 0
        state.node_features[batch_indices, pomo_indices, :, 5] = 0

        # Update the last visited city
        state.node_features[batch_indices, pomo_indices, action.unsqueeze(2), 5] = 1

        return state

    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the solutions based on the selected actions.
        Actions are given in a tuple format, where the first part is the selected node and the second part is the selected class.
        In the case of the TSP, we only need the selected node; this is equivalent to having a single class.
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
        This function is used to update the mask based on the selected actions (cities).
        """
        # There is only one class (selected node) in the TSP, so only take the first part of the action tuple
        action = action[0]

        # Initialize indices for batch and POMO dimensions
        batch_range = torch.arange(state.batch_size, device=state.device)[:, None].expand(state.batch_size, state.pomo_size)
        pomo_range = torch.arange(state.pomo_size, device=state.device)[None, :].expand(state.batch_size, state.pomo_size)

        # Mask the selected city
        state.mask[batch_range, pomo_range, action, :] = float('-inf')
        return state

    def _check_completeness(self, state: State) -> State:
        """
        This function is used to check if the solution is complete.
        """
        # Solution is complete if all cities are visited
        state.is_complete = (state.solutions.size(2) == state.problem_size)
        return state


# %%
# 2) Define the environment, the model, and the trainer
tsp_problem = TSPConstructiveProblem(device=device)

# Now, we define the environment for the TSP (permutation) using a constructive mode
tsp_env = Env(problem=tsp_problem,
              reward=ConstructiveReward(),
              stopping_criteria=ConstructiveStoppingCriteria(),
              device=device)

# Define the model based on 2 node features (2D coordinates)
tsp_model = GTModel(decoder='attention', node_in_dim=6, aux_node=True, logit_clipping=10.0).to(device)
#tsp_model = GCNModel(node_in_dim=6, n_layers=8, decoder='attention', aux_node=True)

# Define the RL training algorithm
tsp_trainer = ConstructiveTrainer(model=tsp_model,
                                  env=tsp_env,
                                  optimizer=torch.optim.Adam(tsp_model.parameters(), lr=5e-4),
                                  device=device)
# %%
# 3) Run training and inference for the Traveling Salesman Problem
train_results = tsp_trainer.train(epochs=1, episodes=1000, problem_size=20, batch_size=64, pomo_size=1, save_freq=-1,
                                  eval_problem_size=20, eval_batch_size=256, baseline_type='mean', verbose=True)



inference_results = tsp_trainer.inference(problem_size=20, batch_size=1, pomo_size=1,
                                          deterministic=True, seed=42, verbose=True)


# %%
# 4) Load the model and run inference
path_to_checkpoint = '../runs/checkpoint.pth'  # <-- Modify this with the saved checkpoint path
tsp_trainer.load_checkpoint(path_to_checkpoint)

inference_results2 = tsp_trainer.inference(problem_size=20, batch_size=1, pomo_size=1,
                                           deterministic=True, seed=42, verbose=True)
