import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import random
from typing import Tuple

import torch
import torch.nn.functional as F

from nco_lib.environment.env import State, Env, ConstructiveStoppingCriteria, ConstructiveReward, ImprovementReward, ImprovementStoppingCriteria
from nco_lib.environment.problem_def import ConstructiveProblem, ImprovementProblem
from nco_lib.models.graph_transformer import GTModel, EdgeInGTModel, EdgeInOutGTModel
from nco_lib.data.data_loader import generate_random_graph
from nco_lib.trainer.trainer import ConstructiveTrainer, ImprovementTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)


# %%
"""
Maximum Cut Problem (MC)
The Maximum Cut problem is a well-known combinatorial optimization problem that consists of finding the partition 
of the nodes of a graph into two sets such that the number of edges between the two sets is maximized.

We will define the MC problem as a constructive problem, where in each step we will assign a node to one of the two subsets.
"""


# %%
# 1) Define the MC constructive problem
class MCConstructiveProblem(ConstructiveProblem):
    def _init_instances(self, state: State) -> State:
        """
        Here the user can define the generation of the instances for the MaxCut problem.
        These instances will be later used to define the node/edge features.
        The state class supports the adjacency matrix as a default data field.
        If any other data is needed, it should be stored in the state.data dictionary.
        The instances will generally have a shape of (batch_size, problem_size, problem_size, n) or (batch_size, problem_size, n), where n is the number of features.
        """
        # In the MaxCut problem (with 0-1 weights), we generate the adjacency matrix as the unique instance information
        # We will use a predefined function to generate a random graph with a given edge-probability (15%)
        state.adj_matrix = generate_random_graph(state.batch_size, state.problem_size, state.seed, edge_prob=0.15, device=device)
        return state

    def _init_solutions(self, state: State) -> State:
        """
        Here the user can define the initialization of the solutions for the MC problem.
        In constructive methods, the solutions are generally initialized empty.
        For the MC we have two options:
        1) Initialize the solutions with zeros (no node assigned to any subset).
        2) Initialize the solutions with random nodes assigned to one of the subsets.
        The solution will generally have a shape of (batch_size, pomo_size, 0~problem_size).
        """

        # Initialize with zeros
        state.solutions = torch.zeros((state.batch_size, state.pomo_size, state.problem_size), device=state.device)

        # Random initialization
        random_init = True
        if random_init:
            batch_indices = torch.arange(state.batch_size, device=state.device)[:, None, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))
            pomo_indices = torch.arange(state.pomo_size, device=state.device)[None, :, None].expand(state.batch_size, state.pomo_size, state.solutions.size(1))

            rand_node_indices = torch.randint(0, state.problem_size, (state.batch_size, state.pomo_size, 1), device=state.device)
            rand_subclass = torch.randint(1, 3, (state.batch_size, state.pomo_size, 1), device=state.device)
            state.solutions[batch_indices, pomo_indices, rand_node_indices] = rand_subclass.float()

            state.data['first_action'] = (rand_node_indices, rand_subclass)
        else:
            state.data['first_action'] = None

        return state

    def _init_features(self, state: State) -> State:
        """
        Here the user can define the initialization of the node/edge features for the MC problem.

        For the MC, we will use the current solutions as node features and the adjacency matrix as edge features.
        """
        # Generate the node features, we will use the three states of the solutions as node features (two classes and one for unassigned)
        state.node_features = F.one_hot(state.solutions.long(), 3).float()

        # Use adjacency matrix as edge features
        state.edge_features = state.adj_matrix.unsqueeze(1).repeat(1, state.pomo_size, 1, 1).unsqueeze(-1)
        return state

    def _init_mask(self, state: State) -> State:
        """
        Here the user can define the initialization of the mask.
        In the MC, the mask will be used to prevent the model from selecting the same node multiple times.
        """
        state.mask = torch.zeros((state.batch_size, state.pomo_size, state.problem_size, 1), device=state.device)

        if state.data['first_action'] is not None:
            first_action = state.data['first_action'][0]
            batch_range = torch.arange(state.batch_size, device=state.device)[:, None, None].expand(state.batch_size, state.pomo_size, 1)
            pomo_range = torch.arange(state.pomo_size, device=state.device)[None, :, None].expand(state.batch_size, state.pomo_size, 1)
            state.mask[batch_range, pomo_range, first_action, :] = float('-inf')

        return state

    def _obj_function(self, state: State) -> torch.Tensor:
        """
        In this function, the user needs to define the objective function for the MC problem.
        This function is called only once the solution is completed.
        """
        # Ininitialize the objective values
        obj_values = torch.zeros((state.batch_size, state.pomo_size), device=device)

        # Convert the solutions to the Ising model
        ising_solutions = state.solutions.clone()
        ising_solutions[ising_solutions == 1] = -1
        ising_solutions[ising_solutions == 2] = 1

        # Calculate the objective values
        for b in range(state.batch_size):
            for p in range(state.pomo_size):
                obj_values[b, p] = (1 / 4) * torch.sum(torch.mul(state.adj_matrix[b], 1 - torch.outer(ising_solutions[b, p], ising_solutions[b, p])))
        return obj_values

    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to define how to update the node/edge features based on the new partial solutions.
        """
        # Update the node features, we will use the three states of the solutions as node features (two classes and one for unassigned)
        state.node_features = F.one_hot(state.solutions.long(), 3).float()
        return state

    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the solutions based on the selected actions.
        Actions are given in a tuple format, where the first part is the selected node and the second part is the selected class.
        In the case of the MC, there are two classes (subsets of nodes) (1 and 2)
        """
        nodes = action[0]
        classes = action[1]

        batch_range = torch.arange(state.batch_size, device=state.device)[:, None].expand(state.batch_size, state.pomo_size)
        pomo_range = torch.arange(state.pomo_size, device=state.device)[None, :].expand(state.batch_size, state.pomo_size)
        state.solutions[batch_range, pomo_range, nodes] = classes.float() + 1
        return state

    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the mask based on the selected actions (cities).
        """
        nodes = action[0]
        batch_range = torch.arange(state.batch_size, device=state.device)[:, None].expand(state.batch_size, state.pomo_size)
        pomo_range = torch.arange(state.pomo_size, device=state.device)[None, :].expand(state.batch_size, state.pomo_size)

        state.mask[batch_range, pomo_range, nodes, :] = float('-inf')
        return state

    def _check_completeness(self, state: State) -> State:
        """
        This function is used to check if the solution is complete.
        """
        state.is_complete = (state.solutions == 0).sum() == 0
        return state


# %%
# 2) Define the environment, the model, and the trainer
mc_problem = MCConstructiveProblem(device=device)

# Now, we define the environment for the MC
mc_env = Env(problem=mc_problem,
             reward=ConstructiveReward(),
             stopping_criteria=ConstructiveStoppingCriteria(),
             device=device)

# Define the model based on edge features (adjacency matrix)
mc_model = EdgeInGTModel(node_in_dim=3, node_out_dim=2, edge_in_dim=1, decoder='attention', aux_node=True,
                         logit_clipping=10).to(device)

# Define the RL training algorithm
mc_trainer = ConstructiveTrainer(model=mc_model,
                                 env=mc_env,
                                 optimizer=torch.optim.Adam(mc_model.parameters(), lr=5e-4),
                                 device=device)
# %%
# 3) Run training and inference for the Maximum Cut Problem (MC)
mc_trainer.inference(problem_size=20, batch_size=100, pomo_size=3, deterministic=True, seed=42, verbose=True)

mc_trainer.train(epochs=10, episodes=1, problem_size=20, batch_size=32, pomo_size=5, eval_problem_size=20,
                 eval_batch_size=256, learn_algo='reinforce', baseline_type='pomo', save_freq=3, save_path_name='maxcut',
                 seed=42, verbose=True)

mc_trainer.inference(problem_size=20, batch_size=100, pomo_size=1, deterministic=True, seed=42, verbose=True)
