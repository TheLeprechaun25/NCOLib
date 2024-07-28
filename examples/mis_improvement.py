import random
from typing import Tuple

import torch
import torch.nn.functional as F

from models.gat import GATModel
from nco_lib.environment.actions import two_opt, bit_flip
from nco_lib.environment.env import State, Env, ConstructiveStoppingCriteria, ConstructiveReward, ImprovementReward, ImprovementStoppingCriteria
from nco_lib.environment.problem_def import ConstructiveProblem, ImprovementProblem
from nco_lib.models.graph_transformer import GTModel, EdgeInGTModel, EdgeInOutGTModel
from nco_lib.data.data_loader import generate_random_graph
from trainer.trainer import ConstructiveTrainer, ImprovementTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)


# %%
"""
Maximum Independent Set Problem (MIS)
The Maximum Independent Set problem is a well-known combinatorial optimization problem that consists of finding the
largest set of nodes in a graph such that no two nodes are adjacent.

We will define the MIS problem as an improvement problem, where in each step we will add or remove a node from the set.
"""


# %%
# 1) Define the MIS improvement problem
class MISImprovementProblem(ImprovementProblem):
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
        state.adj_matrix = generate_random_graph(state.batch_size, state.problem_size, state.seed, edge_prob=0.15, device=device)
        return state

    def _init_solutions(self, state: State) -> State:
        """
        Here the user can define the initialization of the solutions for the MC problem.
        In improvement methods, the solutions are generally initialized as complete solutions, and then improved.

        For the MIS we will initialize the set of nodes greedily, by iteratively adding nodes that are not adjacent to the current set.
        """

        if state.seed is not None:
            random.seed(state.seed)
        # Generate the initial solutions
        solutions = torch.zeros(state.batch_size, state.pomo_size, state.problem_size, device=device)

        # Precompute the neighbors for each node in each graph
        neighbors = [torch.nonzero(state.adj_matrix[b], as_tuple=False) for b in range(state.batch_size)]
        for b in range(state.batch_size):
            for p in range(state.pomo_size):
                available_nodes = set(range(state.problem_size))
                node_neighbors = neighbors[b]
                while available_nodes:
                    node = random.sample(list(available_nodes), 1)[0]
                    # Vectorized check for independent set condition
                    if not torch.any((state.adj_matrix[b, node] == 1) & (solutions[b, p] == 1)):
                        solutions[b, p, node] = 1
                        # Remove the node and its neighbors
                        neighbor_nodes = node_neighbors[node_neighbors[:, 0] == node][:, 1]
                        available_nodes -= {node, *neighbor_nodes.tolist()}
                    else:
                        available_nodes.remove(node)

        state.solutions = solutions
        return state

    def _init_features(self, state: State) -> State:
        """
        Here the user can define the initialization of the node/edge features for the MIS problem.

        For the MIS, we will use the current solutions as node features and the adjacency matrix as edge features.
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
        action = (torch.empty(0), torch.empty(0))  # Empty action for initialization
        return self._update_mask(state, action)

    def _obj_function(self, state: State) -> torch.Tensor:
        """
        In this function, the user needs to define the objective function for the MIS problem.
        This function is called in every step for improvement methods.
        """
        return state.solutions.sum(2).float()

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

        # Use batch matrix multiplication to find if any adjacent node is in the set
        solutions = state.solutions.clone().reshape(state.batch_size*state.pomo_size, state.problem_size)
        adj_matrix = state.adj_matrix.clone().unsqueeze(1).repeat(1, state.pomo_size, 1, 1).reshape(state.batch_size*state.pomo_size, state.problem_size, state.problem_size)
        adjacent_mask = torch.bmm(adj_matrix, solutions.unsqueeze(2).float()).squeeze(2)

        # Nodes that can't be added (any adjacent node is in the set)
        mask = torch.zeros(state.batch_size*state.pomo_size, state.problem_size, device=device)
        masked_index = (adjacent_mask > 0) & (solutions == 0)
        mask[masked_index] = float('-inf')

        # Reshape back to the original shape
        state.mask = mask.reshape(state.batch_size, state.pomo_size, state.problem_size, 1)

        return state

    def _check_completeness(self, state: State) -> State:
        state.is_complete = True
        return state


# %%
# 2) Define the environment, the model, and the trainer
mis_problem = MISImprovementProblem(device=device)

# Now, we define the environment for the MIS problem
mis_env = Env(problem=mis_problem,
              reward=ConstructiveReward(),
              stopping_criteria=ImprovementStoppingCriteria(max_steps=20, patience=3),
              device=device)

# Define the model based on edge features (adjacency matrix)
mis_model = EdgeInGTModel(node_in_dim=2, node_out_dim=1, edge_in_dim=1, decoder='linear', aux_node=False,
                          logit_clipping=10).to(device)

# Define the RL training algorithm
mis_trainer = ImprovementTrainer(model=mis_model,
                                 env=mis_env,
                                 optimizer=torch.optim.Adam(mis_model.parameters(), lr=5e-4),
                                 device=device)
# %%
# 3) Run training and inference for the Maximum Cut Problem (MC)
mis_trainer.inference(problem_size=20, batch_size=100, pomo_size=3, deterministic=True, seed=42, verbose=True)
mis_trainer.train(epochs=10, episodes=10, problem_size=20, batch_size=32, pomo_size=1, eval_problem_size=20,
                  eval_batch_size=256, baseline_type='pomo', save_freq=10, save_path='', seed=42, verbose=True)
mis_trainer.inference(problem_size=20, batch_size=100, pomo_size=3, deterministic=True, seed=42, verbose=True)
