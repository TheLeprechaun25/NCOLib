import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import random
from typing import Tuple

import torch
from nco_lib.environment.env import State, Env, HeatmapReward, HeatmapStoppingCriteria
from nco_lib.environment.problem_def import HeatmapProblem
from nco_lib.models.graph_transformer import EdgeInGTModel
from nco_lib.data.data_loader import generate_random_graph
from nco_lib.trainer.trainer import HeatmapTrainer


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

We will define the MC problem as a heatmap problem, where the model outputs a heatmap that will be decoded into a solution.
Also known as a non-auto-regressive approach.
"""


# %%
# 1) Define the MC heatmap problem
class MCHeatmapProblem(HeatmapProblem):
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

    def _init_features(self, state: State) -> State:
        """
        Here the user can define the initialization of the node/edge features for the MC problem.

        For the MC, we will use the current solutions as node features and the adjacency matrix as edge features.
        """
        # Generate the node features, we will use ones (use pomo_size = 1 for heatmap)
        state.node_features = torch.ones(state.batch_size, 1, state.problem_size, 1, device=device)

        # Use adjacency matrix as edge features
        state.edge_features = state.adj_matrix.unsqueeze(1).unsqueeze(-1)
        return state

    def _decode_solutions(self, state: State, logits: torch.Tensor) -> Tuple[State, torch.Tensor]:
        """
        This function is unique for the heatmap-based approaches.
        Once the model outputs an action,
        """
        log_p = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = log_p.exp()
        probs = probs.reshape(state.batch_size*state.pomo_size*state.problem_size, -1)
        log_probs = []
        solutions = []
        for p in range(state.data['n_rollouts']):
            if state.data['deterministic'] and p == 0:
                sol = probs.argmax(dim=-1).reshape(state.batch_size*state.pomo_size, state.problem_size)
            else:
                sol = probs.multinomial(1).reshape(state.batch_size*state.pomo_size, state.problem_size)

            sol[:, 0] = 1  # set the first node to 1 always
            solutions.append(sol)
            log_probs.append(log_p.gather(2, sol.unsqueeze(-1)).squeeze(-1))

        log_probs = torch.stack(log_probs, dim=1)

        # Average log-probabilities over nodes except the first one,
        # which is fixed to 1 and thus provides no learning signal
        log_probs_mean = log_probs[:, :, 1:].mean(dim=2)
        #log_probs_mean = log_probs.mean(dim=2)  # For other problems not fixing the first node

        state.solutions = torch.stack(solutions, dim=1) # batch_size, n_rollouts, problem_size
        return state, log_probs_mean

    def _obj_function(self, state: State) -> torch.Tensor:
        """
        In this function, the user needs to define the objective function for the MC problem.
        This function is called only once the solution is completed.
        """
        # Convert the solutions to the Ising model
        ising_solutions = 2 * state.solutions.clone() - 1

        # Calculate the objective values
        # 1) Compute outer products:
        outer_solutions = ising_solutions.unsqueeze(-1) * ising_solutions.unsqueeze(-2)
        # 2) Compute (1 - outer_solutions), shape (B, n_rollouts, N, N)
        diff_matrix = 1.0 - outer_solutions
        # 3) Multiply by adj_matrix (broadcast over n_rollouts dimension):
        product = diff_matrix * state.adj_matrix.unsqueeze(1)
        # 4) Sum over the last two dimensions (N, N), then multiply by 1/4
        obj_values = 0.25 * product.sum(dim=(-1, -2))
        return obj_values


# %%
# 2) Define the environment, the model, and the trainer
mc_problem = MCHeatmapProblem(device=device)

# Now, we define the environment for the MC
mc_env = Env(problem=mc_problem,
             reward=HeatmapReward(),
             stopping_criteria=HeatmapStoppingCriteria(),
             device=device)

# Define the model based on edge features (adjacency matrix)
mc_model = EdgeInGTModel(node_in_dim=1, node_out_dim=2, edge_in_dim=1, decoder='attention', aux_node=True,
                         logit_clipping=10).to(device)

# Define the RL training algorithm
mc_trainer = HeatmapTrainer(model=mc_model,
                            env=mc_env,
                            optimizer=torch.optim.AdamW(mc_model.parameters(), lr=1e-4),
                            device=device)
# %%
# 3) Run training and inference for the Maximum Cut Problem (MC)
mc_trainer.inference(problem_size=20, batch_size=100, n_rollouts=10, deterministic=True, seed=42, verbose=True)

mc_trainer.train(epochs=10, episodes=100, problem_size=20, batch_size=128, n_rollouts=100, eval_problem_size=20,
                 eval_batch_size=256, learn_algo='reinforce', save_freq=-1, save_path_name='maxcut',
                 seed=42, verbose=True)

mc_trainer.inference(problem_size=20, batch_size=100, n_rollouts=10, deterministic=True, seed=42, verbose=True)
