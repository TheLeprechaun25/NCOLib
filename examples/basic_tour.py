import random
import torch
import torch.nn.functional as F

from nco_lib.environment.actions import insert, two_opt, swap, bit_flip
from nco_lib.environment.env import State, Env, ConstructiveStoppingCriteria, ConstructiveReward, ImprovementReward, \
    ImprovementStoppingCriteria
from nco_lib.environment.problem_def import ConstructiveProblem, ImprovementProblem, Problem
from nco_lib.models.gnn_model import GTModel, EdgeInGTModel, EdgeInOutGTModel, EdgeOutGTModel
from nco_lib.data.data_loader import generate_random_graph
from nco_lib.trainer import ConstructiveTrainer, ImprovementTrainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DefaultProblem(ConstructiveProblem):
    def _init_instances(self, state: State) -> State:
        # Set the random seed
        if state.seed is not None:
            torch.manual_seed(state.seed)
        # Generate needed features here:
        #state.needed_feature =
        return state

    def _init_solutions(self, state: State) -> State:
        # Initialize the solutions here:
        #state.solutions =
        return state

    def _init_features(self, state: State) -> State:
        # Initialize the features here:
        #state.node_features =
        #state.edge_features =
        return state

    def _init_mask(self, state: State) -> State:
        # Initialize the mask here:
        #state.mask =
        return state

    def _obj_function(self, state: State) -> torch.Tensor:
        # Compute the objective function here:
        #obj_value =
        return obj_value

    def _update_features(self, state: State) -> State:
        # Update the features here:
        #state.node_features =
        #state.edge_features =
        return state

    def _update_solutions(self, state: State, action: torch.Tensor) -> State:
        # Update the solutions with the action here:
        #state.solutions =
        return state

    def _update_mask(self, state: State, action: torch.Tensor) -> State:
        # Update the mask with the action here:
        #state.mask =
        return state

    def _check_completeness(self, state: State) -> State:
        # Check if the solution is complete here:
        #state.is_complete =
        return state


example1_0 = False  # TSP constructive
example1_1 = False  # TSP improvement
example1_2 = False  # TSP heatmap
example2 = False  # MC constructive
example3 = True  # MIS improvement

# Example 1: Inference for the Traveling Salesman Problem (TSP)
if example1_0:
    print("\nExample 1.0: Inference and training a constructive model for the Traveling Salesman Problem (TSP)")
    # First define a TSP problem class

    class TSPConstructiveProblem(ConstructiveProblem):
        def _init_instances(self, state: State) -> State:
            if state.seed is not None:
                torch.manual_seed(state.seed)
            # Generate the city coordinates
            state.coords = torch.rand(state.batch_size, state.problem_size, 2, device=state.device)
            # Compute the Euclidean distance between cities
            state.distances = torch.cdist(state.coords, state.coords)
            return state

        def _init_solutions(self, state: State) -> State:
            state.solutions = torch.zeros((state.batch_size, 0), device=state.device)
            return state

        def _init_features(self, state: State) -> State:
            return self._update_features(state)  # Init features equal to the update every construction step

        def _init_mask(self, state: State) -> State:
            state.mask = torch.zeros((state.batch_size, state.problem_size, 1), device=state.device)
            return state

        def _obj_function(self, state: State) -> torch.Tensor:
            gathering_index = state.solutions.unsqueeze(2).expand(state.batch_size, state.problem_size, 2).long()
            ordered_seq = state.coords.gather(dim=1, index=gathering_index)
            rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()  # shape: (batch, N)
            travel_distances = segment_lengths.sum(1)  # shape: (batch)
            return -travel_distances  # minimize the total distance  -> maximize the negative distance

        def _update_features(self, state: State) -> State:
            selected = torch.zeros(state.batch_size, state.problem_size, 2, device=device)
            selected[torch.arange(state.batch_size), :, 0] = 1

            # Prepare indices for gathering
            batch_indices = torch.arange(state.batch_size, device=device)[:, None].expand(-1, state.solutions.size(1))

            # Mark visited cities
            selected[batch_indices, state.solutions.long(), 0] = 0
            selected[batch_indices, state.solutions.long(), 1] = 1

            state.node_features = torch.cat([state.coords, selected], dim=-1)
            # Optionally, use the distances as edge features. But use an edge-based model!
            # state.edge_features = state.distances
            return state

        def _update_solutions(self, state: State, action: torch.Tensor) -> State:
            # Append the selected city to the solution
            state.solutions = torch.cat([state.solutions, action.unsqueeze(1)], dim=1)
            return state

        def _update_mask(self, state: State, action: torch.Tensor) -> State:
            # Mask the selected city
            batch_range = torch.arange(state.batch_size, device=state.device)
            state.mask[batch_range, action, :] = float('-inf')
            return state

        def _check_completeness(self, state: State) -> State:
            # Solution is complete if all cities are visited
            state.is_complete = (state.solutions.size(1) == state.problem_size)
            return state


    tsp_problem = TSPConstructiveProblem(device=device)

    # Now, we define the environment for the TSP (permutation) using a constructive mode
    tsp_env = Env(problem=tsp_problem,
                  reward=ConstructiveReward(),
                  stopping_criteria=ConstructiveStoppingCriteria(),
                  device=device)

    # Define the model based on 2 node features (2D coordinates)
    tsp_model = GTModel(node_in_dim=4).to(device)

    # Define the RL training algorithm
    tsp_trainer = ConstructiveTrainer(model=tsp_model,
                                      env=tsp_env,
                                      optimizer=torch.optim.Adam(tsp_model.parameters(), lr=5e-4),
                                      device=device)

    # Run inference on the model to get the final state and reward
    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)

    # Training for the Traveling Salesman Problem (TSP)
    tsp_trainer.train(epochs=10, episodes=100, problem_size=20, batch_size=32, verbose=True)

    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)

if example1_1:
    print("\nExample 1.1: Inference and training an improvement model for the Traveling Salesman Problem (TSP)")

    class TSPImprovementProblem(ImprovementProblem):
        def _init_instances(self, state: State) -> State:
            if state.seed is not None:
                torch.manual_seed(state.seed)
            # Generate the city coordinates
            state.coords = torch.rand(state.batch_size, state.problem_size, 2, device=state.device)
            # Compute the Euclidean distance between cities
            state.distances = torch.cdist(state.coords, state.coords)
            return state

        def _init_solutions(self, state: State) -> State:
            if state.seed is not None:
                torch.manual_seed(state.seed)
            state.solutions = torch.stack(
                [torch.randperm(state.problem_size, device=state.device) for _ in range(state.batch_size)])

            sets = torch.rand(state.batch_size, state.problem_size, device=state.device).argsort().long()
            rec = torch.zeros(state.batch_size, state.problem_size, device=state.device).long()
            index = torch.zeros(state.batch_size, 1, device=state.device).long()

            for i in range(state.problem_size - 1):
                rec.scatter_(1, sets.gather(1, index + i), sets.gather(1, index + i + 1))
            rec.scatter_(1, sets[:, -1].view(-1, 1), sets.gather(1, index))
            state.solutions = rec
            return state

        def _init_features(self, state: State) -> State:
            return self._update_features(state)

        def _init_mask(self, state: State) -> State:
            # Mask the diagonal elements.
            mask = torch.zeros((state.batch_size, state.problem_size, state.problem_size, 1), device=state.device)
            row_indices = torch.arange(state.problem_size, device=state.device)
            mask[:, row_indices, row_indices, :] = -float('inf')
            # Reshape the mask to (batch_size, problem_size^2, 1)
            state.mask = mask.reshape(state.batch_size, -1, 1)
            return state

        def _obj_function(self, state: State) -> torch.Tensor:
            coor_next = state.coords.gather(1, state.solutions.long().unsqueeze(-1).expand(*state.solutions.size(), 2))
            travel_distances = (state.coords - coor_next).norm(p=2, dim=2).sum(1)
            return -travel_distances

        def _update_features(self, state: State) -> State:
            # Use the 2D coordinates as node features
            state.node_features = torch.ones(state.batch_size, state.problem_size, 1, device=device)  # state.coords
            # Use edge features to encode solutions (edges between selected nodes)
            indices = torch.arange(state.problem_size, device=device).unsqueeze(0).repeat(state.batch_size, 1)
            # Initialize edge solutions tensor
            edge_solutions = torch.zeros(state.batch_size, state.problem_size, state.problem_size, 1,
                                         dtype=torch.float32, device=device)

            # Update edge solutions using advanced indexing
            edge_solutions[torch.arange(state.batch_size, device=device).unsqueeze(1), indices, state.solutions, :] = 1
            # Make the edge solutions symmetric
            edge_solutions = edge_solutions + edge_solutions.permute(0, 2, 1, 3)
            # One-hot encoding of the edge solutions
            edge_solutions = F.one_hot(edge_solutions.squeeze(-1).long(), 2).float()

            state.edge_features = torch.cat([state.distances.unsqueeze(-1), edge_solutions], dim=-1)
            return state

        def _update_solutions(self, state: State, action: torch.Tensor) -> State:
            state.solutions = two_opt(state.solutions, action)
            return state

        def _update_mask(self, state: State, action: torch.Tensor) -> State:
            return state

        def _check_completeness(self, state: State) -> State:
            state.is_complete = True
            return state


    tsp_problem = TSPImprovementProblem(device=device)

    # Now, we define the environment for the TSP (permutation) using a constructive mode
    tsp_env = Env(problem=tsp_problem,
                  reward=ImprovementReward(),
                  stopping_criteria=ImprovementStoppingCriteria(max_steps=20, patience=3),
                  device=device)

    # Define the model based on 2 node features (2D coordinates)
    tsp_model = EdgeInOutGTModel(node_in_dim=1, edge_in_dim=3, edge_out_dim=1).to(device)

    # Define the RL training algorithm
    tsp_trainer = ImprovementTrainer(model=tsp_model,
                                     env=tsp_env,
                                     optimizer=torch.optim.Adam(tsp_model.parameters(), lr=5e-4),
                                     device=device)

    # Run inference on the model to get the final state and reward
    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)

    # Training for the Traveling Salesman Problem (TSP)
    tsp_trainer.train(epochs=10, episodes=100, problem_size=20, batch_size=128, update_freq=10, gamma=0.95, verbose=True)

    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    tsp_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)

if example1_2:
    print("\nExample 1.2: Inference and training a heatmap-based Non-AutoRegressive model for the Traveling Salesman Problem (TSP)")

    # TODO



# Example 2: Inference and training a constructive model for the Maximum Cut Problem (MC)
if example2:
    print("\nExample 2: Inference and training a constructive model for the Maximum Cut Problem (MC)")

    class MCConstructiveProblem(ConstructiveProblem):
        def _init_instances(self, state: State) -> State:
            state.adj_matrix = generate_random_graph(state.batch_size, state.problem_size, state.seed, edge_prob=0.15, device=device)
            return state

        def _init_solutions(self, state: State) -> State:
            state.solutions = torch.zeros((state.batch_size, state.problem_size), device=state.device)
            return state

        def _init_features(self, state: State) -> State:
            return self._update_features(state)  # Init features equal to the update every construction step

        def _init_mask(self, state: State) -> State:
            state.mask = torch.zeros((state.batch_size, state.problem_size, 2), device=state.device)
            return state

        def _obj_function(self, state: State) -> torch.Tensor:
            batch_size, N = state.solutions.shape
            obj_values = torch.zeros(batch_size, device=device)
            ising_solutions = state.solutions.clone()
            ising_solutions[ising_solutions == 1] = -1
            ising_solutions[ising_solutions == 2] = 1
            adj_matrix = state.edge_features.squeeze(-1)
            for b in range(batch_size):
                obj_values[b] = (1 / 4) * torch.sum(
                    torch.mul(adj_matrix[b], 1 - torch.outer(ising_solutions[b], ising_solutions[b])))
            return obj_values

        def _update_features(self, state: State) -> State:
            # Generate the node features, we will use the three states of the solutions as node features (two classes and one for unassigned)
            state.node_features = F.one_hot(state.solutions.long(), 3).float()
            # Use adjacency matrix as edge features
            state.edge_features = state.adj_matrix.unsqueeze(-1)
            return state

        def _update_solutions(self, state: State, action: torch.Tensor) -> State:
            classes = action % 2
            nodes = action // 2
            batch_range = torch.arange(state.batch_size, device=state.device)
            state.solutions[batch_range, nodes] = classes.float() + 1
            return state

        def _update_mask(self, state: State, action: torch.Tensor) -> State:
            nodes = action // 2
            batch_range = torch.arange(state.batch_size, device=state.device)
            state.mask[batch_range, nodes, :] = float('-inf')
            return state

        def _check_completeness(self, state: State) -> State:
            state.is_complete = (state.solutions == 0).sum() == 0
            return state


    mc_problem = MCConstructiveProblem(device=device)

    # Now, we define the environment for the MC
    mc_env = Env(problem=mc_problem,
                 reward=ConstructiveReward(),
                 stopping_criteria=ConstructiveStoppingCriteria(),
                 device=device)

    # Define the model based on edge features (adjacency matrix)
    mc_model = EdgeInGTModel(node_in_dim=3, node_out_dim=2, edge_in_dim=1).to(device)

    # Define the RL training algorithm
    mc_trainer = ConstructiveTrainer(model=mc_model, env=mc_env, optimizer=torch.optim.Adam(mc_model.parameters()),
                                     device=device)

    # Run inference on the model to get the final state and reward
    mc_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    mc_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)

    # Training for the Maximum Cut Problem (MC)
    mc_trainer.train(epochs=10, episodes=10, problem_size=20, batch_size=32, save_freq=10, save_path=None, verbose=True)

    mc_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    mc_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)

# Example 3: Inference and training an improvement model for the Maximum Independent Set Problem (MIS)
if example3:
    print("\nExample 3: Inference and training an improvement model for the Maximum Independent Set Problem (MIS)")

    class MISImprovementProblem(ImprovementProblem):
        def _init_instances(self, state: State) -> State:
            state.adj_matrix = generate_random_graph(state.batch_size, state.problem_size, state.seed, edge_prob=0.15, device=device)
            return state

        def _init_solutions(self, state: State) -> State:
            if state.seed is not None:
                random.seed(state.seed)
            # Generate the initial solutions
            solutions = torch.zeros(state.batch_size, state.problem_size, device=device)

            # Precompute the neighbors for each node in each graph
            neighbors = [torch.nonzero(state.adj_matrix[b], as_tuple=False) for b in range(state.batch_size)]
            for b in range(state.batch_size):
                available_nodes = set(range(state.problem_size))
                node_neighbors = neighbors[b]
                while available_nodes:
                    node = random.sample(list(available_nodes), 1)[0]
                    # Vectorized check for independent set condition
                    if not torch.any((state.adj_matrix[b, node] == 1) & (solutions[b] == 1)):
                        solutions[b, node] = 1
                        # Remove the node and its neighbors
                        neighbor_nodes = node_neighbors[node_neighbors[:, 0] == node][:, 1]
                        available_nodes -= {node, *neighbor_nodes.tolist()}
                    else:
                        available_nodes.remove(node)

            state.solutions = solutions
            return state

        def _init_features(self, state: State) -> State:
            return self._update_features(state)

        def _init_mask(self, state: State) -> State:
            return self._update_mask(state, None)

        def _obj_function(self, state: State) -> torch.Tensor:
            return state.solutions.sum(1).float()

        def _update_features(self, state: State) -> State:
            # Generate the node weights, we will use a weight of 1 for all nodes
            state.node_features = F.one_hot(state.solutions.long(), 2).float()
            # Use adjacency matrix as edge features
            state.edge_features = state.adj_matrix.unsqueeze(-1)
            return state

        def _update_solutions(self, state: State, action: torch.Tensor) -> State:
            state.solutions = bit_flip(state.solutions, action)
            return state

        def _update_mask(self, state: State, action: torch.Tensor or None) -> State:
            # Use batch matrix multiplication to find if any adjacent node is in the set
            adjacent_mask = torch.bmm(state.edge_features.squeeze(-1), state.solutions.unsqueeze(2).float()).squeeze(2)

            # Nodes that can't be added (any adjacent node is in the set)
            mask = torch.zeros(state.batch_size, state.problem_size, device=device)
            masked_index = (adjacent_mask > 0) & (state.solutions == 0)
            mask[masked_index] = float('-inf')
            state.mask = mask.unsqueeze(-1)
            return state

        def _check_completeness(self, state: State) -> State:
            state.is_complete = True
            return state


    mis_problem = MISImprovementProblem(device=device)

    mis_env = Env(problem=mis_problem,
                  reward=ImprovementReward(),
                  stopping_criteria=ImprovementStoppingCriteria(max_steps=20, patience=3),
                  device=device)

    # Define the model based on edge features (adjacency matrix)
    mis_model = EdgeInGTModel(node_in_dim=2, node_out_dim=1, edge_in_dim=1).to(device)

    # Define the RL training algorithm
    mis_trainer = ImprovementTrainer(model=mis_model,
                                     env=mis_env,
                                     optimizer=torch.optim.Adam(mis_model.parameters(), lr=1e-3),
                                     device=device)

    # Run inference on the model to get the final state and reward
    mis_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    mis_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)

    # Training for the Maximum Cut Problem (MC)
    mis_trainer.train(epochs=10, episodes=10, problem_size=20, batch_size=32, save_freq=10, save_path=None, verbose=True)

    # Run inference on the model to get the final state and reward
    mis_trainer.inference(problem_size=20, batch_size=100, deterministic=True, seed=42, verbose=True)
    mis_trainer.inference(problem_size=20, batch_size=100, deterministic=False, seed=42, verbose=True)
