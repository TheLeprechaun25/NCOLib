import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import random
from typing import Tuple

import torch
import torch.nn.functional as F

from nco_lib.environment.actions import bit_flip
from nco_lib.environment.env import State, Env, ImprovementReward, ImprovementStoppingCriteria
from nco_lib.environment.problem_def import ImprovementProblem
from nco_lib.environment.memory import select_memory
from nco_lib.models.graph_transformer import EdgeInGTModel
from nco_lib.data.data_loader import generate_random_graph
from nco_lib.trainer.trainer import ImprovementTrainer


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

We will define the MC problem as an improvement problem, where in each step we will add or remove a node from one of the sets to the other.
"""


# %%
# 1) Define the MC improvement problem
class MCImprovementProblem(ImprovementProblem):
    def _init_instances(self, state: State) -> State:
        """
        Here the user can define the generation of the instances for the MIS problem.
        These instances will be later used to define the node/edge features.
        The state class supports the adjacency matrix as a default data field.
        If any other data is needed, it should be stored in the state.data dictionary.
        The instances will generally have a shape of (batch_size, problem_size, problem_size, n) or (batch_size, problem_size, n), where n is the number of features.
        """
        # In the MIS problem, we generate the adjacency matrix as the unique instance information
        # We will use a predefined function to generate a random graph with a given edge-probability (15%)
        state.adj_matrix = generate_random_graph(state.batch_size, state.problem_size, state.seed, edge_prob=0.15, device=self.device)
        return state

    def _init_solutions(self, state: State) -> State:
        """
        Here the user can define the initialization of the solutions for the MC problem.
        In improvement methods, the solutions are generally initialized as complete solutions, and then improved.

        For the MC we will initialize the set of nodes greedily, by iteratively adding nodes that are not adjacent to the current set.
        """

        if state.seed is not None:
            random.seed(state.seed)

        # Generate the initial solutions
        state.solutions = torch.randint(0, 2, (state.batch_size, state.pomo_size, state.problem_size), device=self.device)
        return state

    def _init_features(self, state: State) -> State:
        """
        Here the user can define the initialization of the node/edge features for the MIS problem.

        For the MC, we will use the current solutions as node features and the adjacency matrix as edge features.
        """
        # Generate the node features, we will use the two states of the solutions as node features (in or out of the set)
        state.node_features = F.one_hot(state.solutions.long(), 2).float()

        # Use adjacency matrix as edge features
        state.edge_features = state.adj_matrix.unsqueeze(1).repeat(1, state.pomo_size, 1, 1).unsqueeze(-1)
        return state

    def _init_mask(self, state: State) -> State:
        """
        Here the user can define the initialization of the mask.
        In the improvement version of MIS, and typically in most of the improvement methods,
        the _init_mask function and the _update_mask function are the same, so we will call it from here.
        """
        # No mask
        state.mask = torch.ones((state.batch_size, state.pomo_size, state.problem_size, 1), device=self.device)
        return state

    def _obj_function(self, state: State) -> torch.Tensor:
        """
        In this function, the user needs to define the objective function for the MIS problem.
        This function is called in every step for improvement methods.
        """
        obj_value = torch.zeros(state.batch_size, state.pomo_size, device=self.device)
        ising_solutions = 2 * state.solutions - 1
        for b in range(state.batch_size):
            for p in range(state.pomo_size):
                obj_value[b, p] = (1 / 4) * torch.sum(torch.mul(state.adj_matrix[b], 1 - torch.outer(ising_solutions[b, p], ising_solutions[b, p])))

        return obj_value

    def _update_features(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to define how to update the node/edge features based on the new partial solutions.
        """
        # Only node features need to be updated, edge features are static
        state.node_features = F.one_hot(state.solutions.long(), 2).float()

        return state

    def _update_solutions(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the solutions based on the selected actions.
        Actions are given in a tuple format, where the first part is the selected node and the second part is the selected class.
        In our case (improvement for MIS), we only have one class (select a node and flip its value) so we will only use the first part of the action.
        """
        action = action[0]

        # Update the solutions by flipping the selected node
        state.solutions = bit_flip(state.solutions, action)
        return state

    def _update_mask(self, state: State, action: Tuple[torch.Tensor, torch.Tensor]) -> State:
        """
        This function is used to update the mask based on the selected actions (flipped nodes).
        """
        return state

    def _check_completeness(self, state: State) -> State:
        state.is_complete = True
        return state


# %%
# 2) Define the environment, the model, and the trainer
problem_size = 20
batch_size = 32
pomo_size = 1
mc_problem = MCImprovementProblem(device=device)

memory_type = 'marco_shared'  # 'marco_shared', 'marco_individual or 'last_action' or 'none'
mc_memory = select_memory(memory_type=memory_type,
                          mem_aggr='linear',
                          state_dim=problem_size,
                          action_dim=problem_size,
                          batch_size=batch_size,
                          pomo_size=pomo_size,
                          repeat_punishment=2.0,
                          device=device)

# Now, we define the environment for the MC problem
mc_env = Env(problem=mc_problem,
             reward=ImprovementReward(positive_only=True, normalize=False),
             stopping_criteria=ImprovementStoppingCriteria(max_steps=20, patience=20),
             memory=mc_memory,
             device=device)

# Compute the input dimension for the model based on the node features and the memory type
node_in_dim = 2  # Two classes for the nodes (in or out of the set)
if memory_type == 'last_action':
    node_in_dim += 1  # Add one dimension for the last time the action was selected
elif memory_type.startswith('marco'):
    node_in_dim += 2  # Add two dimensions for the memory information of marco

# Define the model based on edge features (adjacency matrix)
mc_model = EdgeInGTModel(node_in_dim=node_in_dim, node_out_dim=1, edge_in_dim=1, decoder='linear', aux_node=False,
                         logit_clipping=10).to(device)

# Define the RL training algorithm
mc_trainer = ImprovementTrainer(model=mc_model,
                                env=mc_env,
                                optimizer=torch.optim.Adam(mc_model.parameters(), lr=5e-4),
                                device=device)

# %%
# 3) Run training and inference for the Maximum Cut Problem (MC)
mc_trainer.inference(problem_size=problem_size, batch_size=batch_size, pomo_size=pomo_size, deterministic=True, seed=42, verbose=True)
mc_trainer.train(epochs=10, episodes=10, problem_size=problem_size, batch_size=batch_size, pomo_size=pomo_size, eval_problem_size=problem_size,
                 eval_batch_size=batch_size, baseline_type='mean', save_freq=10, save_path='', seed=42, verbose=True)
mc_trainer.inference(problem_size=problem_size, batch_size=batch_size, pomo_size=pomo_size, deterministic=True, seed=42, verbose=True)
