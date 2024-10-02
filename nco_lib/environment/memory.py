from abc import abstractmethod

import torch


class Memory:
    def __init__(self, state_dim, action_dim, device='cpu'):
        """
        Memory that saves State-Action (key-value) pairs and returns the average value of the k-nearest neighbours of a state
        state_dim: int: The dimension of the state (key) space
        action_dim: int: The dimension of the action (value) space
        """
        self.state_dim = state_dim  # Key_dim
        self.action_dim = action_dim  # Value_dim
        self.repeat_punishment = None
        self.k = None
        self.device = device

    @abstractmethod
    def save_in_memory(self, state, action):
        """
        Stores the state-action pair in the memory
            state (key): torch.Tensor: The state of the problem
            action (value): torch.Tensor: The action (node/edge) that was taken

        Returns:

        """
        pass

    @abstractmethod
    def get_knn(self, state, k):
        """
        Get the average value of the k-nearest neighbors (keys) of the query state
            state: torch.Tensor: The state (current solution) of the problem to be compared with the memory
            k: int: The number of nearest neighbors to consider

        Returns:
            nearest_actions: torch.Tensor: The average value of the k-nearest keys
            revisited: torch.Tensor: The number of times the same state has been visited
            avg_similarity: np.array: The average similarity of the k-nearest neighbors
            max_similarity: np.array: The maximum similarity of the k-nearest neighbors

        """
        pass

    @abstractmethod
    def clear_memory(self):
        """
        Clears the memory
        """
        pass



class MarcoMemory(Memory):
    def __init__(self, state_dim, action_dim, n_memories, batch_size, pomo_size, memory_aggr, k=10, memory_size=100000, repeat_punishment=0.0, device='cpu'):
        super().__init__(state_dim, action_dim, device)
        """
        Memory that saves State-Action pairs and returns the average value of the k-nearest neighbours of a state
        Key: Solution state of the problem
        Value: One-hot encoding of the performed action
        """
        self.memory_size = memory_size
        self.n_memories = n_memories
        self.batch_size = batch_size
        self.pomo_size = pomo_size
        self.memory_aggr = memory_aggr
        self.k = k
        self.repeat_punishment = repeat_punishment

        # Initialize memories and index
        self.state_memories = [torch.zeros((0, self.state_dim)) for _ in range(self.n_memories)]
        self.action_memories = [torch.zeros((0, self.action_dim, 2)) for _ in range(self.n_memories)]

        self.used_memory = 0

    def save_in_memory(self, state, action):
        """
        Stores the state-action pair in the memory
            state (key): torch.Tensor: The state (current solution) of the problem
            action (value): torch.Tensor: The action (node/edge) that was taken

        Returns:

        """
        assert state.batch_size == self.batch_size, "Batch size of state and memory do not match"
        assert state.pomo_size == self.pomo_size, "POMO size of state and memory do not match"

        batch_pomo_size = state.batch_size * state.pomo_size
        batch_pomo_range = torch.arange(batch_pomo_size)

        # Select the first part. TODO: marco for multiclass problems
        action = action[0].reshape(batch_pomo_size).cpu()

        key = state.solutions.reshape(batch_pomo_size, self.state_dim)

        # Get the current values in solutions
        cur_values = key[batch_pomo_range, action]

        # Double the actions so that we can distinguish between 0->1 and 1->0
        double_action = action.clone()
        double_action[cur_values == 1] += self.state_dim

        # Perform one-hot encoding
        one_hot_actions = torch.nn.functional.one_hot(double_action, num_classes=2*self.state_dim).float()
        one_hot_actions = one_hot_actions.view(batch_pomo_size, 2, self.state_dim)
        one_hot_actions = one_hot_actions.transpose(1, 2).contiguous().view(batch_pomo_size, self.state_dim, 2)

        for idx in range(self.n_memories):
            # If memory is full, remove the oldest state
            if self.used_memory >= self.memory_size:
                if self.n_memories == 1:
                    self.state_memories[idx] = torch.roll(self.state_memories[idx], -batch_pomo_size, dims=0)
                    self.state_memories[idx][-batch_pomo_size:] = key
                    self.action_memories[idx] = torch.roll(self.action_memories[idx], -batch_pomo_size, dims=0)
                    self.action_memories[idx][-batch_pomo_size:] = one_hot_actions
                else:
                    self.state_memories[idx] = torch.roll(self.state_memories[idx], -1, dims=0)
                    self.state_memories[idx][-1] = key[idx]
                    self.action_memories[idx] = torch.roll(self.action_memories[idx], -1, dims=0)
                    self.action_memories[idx][-1] = one_hot_actions[idx].unsqueeze(0)
            else:
                if self.n_memories == 1:
                    self.state_memories[idx] = torch.vstack([self.state_memories[idx], key])
                    self.action_memories[idx] = torch.vstack([self.action_memories[idx], one_hot_actions])
                else:
                    self.state_memories[idx] = torch.vstack([self.state_memories[idx], key[idx]])
                    self.action_memories[idx] = torch.vstack([self.action_memories[idx], one_hot_actions[idx].unsqueeze(0)])

        if self.n_memories == 1:
            self.used_memory += batch_pomo_size
        else:
            self.used_memory += 1

    def get_knn(self, state, k):
        """
        Get the average value of k-nearest neighbors (keys) of the query state
            query: torch.Tensor: The state (current solution) of the problem to be compared with the memory
            k: int: The number of nearest neighbors to consider

        Returns:
            nearest_actions: torch.Tensor: The average value of the k-nearest keys
            revisited: torch.Tensor: The number of times the same state has been visited
            avg_similarity: np.array: The average similarity of the k-nearest neighbors
            max_similarity: np.array: The maximum similarity of the k-nearest neighbors

        """
        assert state.batch_size == self.batch_size, "Batch size of state and memory do not match"
        assert state.pomo_size == self.pomo_size, "POMO size of state and memory do not match"

        batch_pomo_size = state.batch_size * state.pomo_size
        k = self.used_memory if k > self.used_memory else k

        query = state.solutions.clone()
        query = query.reshape(batch_pomo_size, self.state_dim).float()

        nearest_actions = torch.zeros((batch_pomo_size, self.action_dim, 2))

        avg_similarity = torch.zeros(batch_pomo_size)
        max_similarity = torch.zeros(batch_pomo_size)
        revisited = torch.zeros(batch_pomo_size)

        # Get k nearest states
        for idx in range(self.n_memories):
            if self.n_memories != 1:  # Multiple memories
                inner_products = torch.mm(query[idx:idx+1], self.state_memories[idx].t())  # Result is pomo_size * used_memory
                similarity, indices = torch.topk(inner_products, k, largest=True, sorted=True)
                # similarity.shape = (1, k) and indices.shape = (1, k)

                revisited[idx] = (similarity == self.state_dim).sum()
                avg_similarity[idx] = torch.mean(similarity)
                max_similarity[idx] = similarity[:, 0]

                nearest_acts = self.action_memories[idx][indices.flatten(), :].reshape(indices.shape + (self.state_dim, 2))
                # nearest_acts.shape = (1, k, state_dim, 2)
                # Aggregate among k neighbors: Weighted based on similarity.
                if self.memory_aggr == 'sum':  # No weighting
                    nearest_actions[idx] = torch.sum(nearest_acts, dim=1)
                elif self.memory_aggr == 'linear':  # Linear weighted sum
                    # similarity from [-N, N] --> [0, N] --> [0, 1]
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions[idx] = torch.sum(nearest_acts * sim[:, :, None, None], dim=1)
                elif self.memory_aggr == 'exp':  # Exponential weighted sum
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions[idx] = torch.sum(nearest_acts * (torch.exp(torch.log(torch.tensor(2)) * (sim[:, :, None, None])) - 1), dim=1)
                    # nearest_actions.shape = (batch_size, state_dim, 2)
            else:  # Single memory
                inner_products = torch.mm(query, self.state_memories[idx].t())  # Result is pomo_size * used_memory
                similarity, indices = torch.topk(inner_products, k, largest=True, sorted=True)
                # similarity.shape = (batch_size, k) and indices.shape = (batch_size, k)

                revisited = (similarity == self.state_dim).sum(axis=1)
                avg_similarity = torch.mean(similarity, dim=1)
                max_similarity = similarity[:, 0]

                nearest_acts = self.action_memories[idx][indices, :].reshape(indices.shape + (self.action_dim, 2))
                # nearest_acts.shape = (batch_size, k, state_dim, 2)
                # Aggregate among k neighbors: Weighted based on similarity.
                if self.memory_aggr == 'sum':  # No weighting
                    nearest_actions = torch.sum(nearest_acts, dim=1)
                elif self.memory_aggr == 'linear':  # Linear weighted sum
                    # similarity from [-N, N] --> [0, N] --> [0, 1]
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions = torch.sum(nearest_acts * sim[:, :, None, None], dim=1)
                elif self.memory_aggr == 'exp':  # Exponential weighted sum
                    sim = (similarity + self.state_dim) / (2 * self.state_dim)
                    nearest_actions = torch.sum(nearest_acts * (torch.exp(torch.log(torch.tensor(2)) * sim[:, :, None, None]) - 1), dim=1)
                    # nearest_actions.shape = (batch_size, state_dim, 2)

        nearest_actions = nearest_actions.reshape(self.batch_size, self.pomo_size, self.state_dim, 2).to(self.device)
        revisited = revisited.view(self.batch_size, self.pomo_size).to(self.device)
        avg_similarity = avg_similarity.reshape(self.batch_size, self.pomo_size).to(self.device)
        max_similarity = max_similarity.reshape(self.batch_size, self.pomo_size).to(self.device)

        return nearest_actions, revisited, avg_similarity, max_similarity

    def clear_memory(self):
        """
        Clears the memory
        """
        self.state_memories = [torch.zeros((0, self.state_dim)) for _ in range(self.n_memories)]
        self.action_memories = [torch.zeros((0, self.action_dim, 2)) for _ in range(self.n_memories)]
        self.used_memory = 0


class LastActionMemory(Memory):
    def __init__(self, state_dim, action_dim, batch_size, pomo_size, repeat_punishment=0.0, device='cpu'):
        super().__init__(state_dim, action_dim, device)
        self.batch_size = batch_size
        self.pomo_size = pomo_size
        self.repeat_punishment = repeat_punishment
        self.device = device

        self.cur_action = torch.zeros(batch_size*pomo_size, action_dim)

        self.steps_since_preformed = -1 * torch.ones(batch_size*pomo_size, action_dim)


    def save_in_memory(self, state, action):
        """
        Stores the last action (value) in the memory
            state: torch.Tensor: The state (current solution) of the problem
            action: torch.Tensor: The action (node/edge) that was taken
        """

        assert state.batch_size == self.batch_size, "Batch size of state and memory do not match"
        assert state.pomo_size == self.pomo_size, "POMO size of state and memory do not match"

        batch_pomo_range = torch.arange(state.batch_size*state.pomo_size)

        # Select the first part. TODO: marco for multiclass problems
        action = action[0].reshape(state.batch_size*state.pomo_size).cpu()

        # Update the current action
        self.cur_action = action

        # Update steps since last performed
        self.steps_since_preformed[self.steps_since_preformed != -1] += 1  # Only update those that have been performed
        self.steps_since_preformed[batch_pomo_range, action] = 1

    def get_knn(self, state, k):
        assert state.batch_size == self.batch_size, "Batch size of state and memory do not match"
        assert state.pomo_size == self.pomo_size, "POMO size of state and memory do not match"

        # check if state.last_action and self.cur_action are the same
        # if they are the same, then the state has been visited before
        # if they are different, then the state has not been visited before
        if state.last_action is not None:
            last_action = state.last_action[0]
            revisited = last_action.reshape(state.batch_size*state.pomo_size) == self.cur_action
            revisited = revisited.view(self.batch_size, self.pomo_size)

        else:
            revisited = torch.zeros((self.batch_size, self.pomo_size), device=self.device)

        # TODO: what to give, the last action or the steps since last performed?
        memory_info = self.steps_since_preformed.clone().reshape(self.batch_size, self.pomo_size, self.state_dim, 1)
        memory_info = memory_info.to(self.device)

        return memory_info, revisited, None, None

    def clear_memory(self):
        """
        Clears the memory
        """
        self.cur_action = torch.zeros((self.batch_size, self.action_dim))
        self.steps_since_preformed = -1 * torch.ones(self.batch_size*self.pomo_size, self.action_dim)


class NoMemory(Memory):
    def __init__(self):
        super().__init__(state_dim=0, action_dim=0)

    def save_in_memory(self, state, action):
        pass

    def get_knn(self, state, k):
        return None, None, None, None

    def clear_memory(self):
        pass


def select_memory(state_dim, action_dim, batch_size, pomo_size, device, memory_type='none', k=10, mem_aggr='linear', repeat_punishment=1.0):
    if memory_type == 'last_action':
        memory = LastActionMemory(state_dim=state_dim,
                                  action_dim=action_dim,
                                  batch_size=batch_size,
                                  pomo_size=pomo_size,
                                  repeat_punishment=repeat_punishment,
                                  device=device)
    elif memory_type == 'marco':
        if memory_type == 'shared':
            n_memories = 1
        else:  # 'individual' or training
            n_memories = batch_size*pomo_size

        memory = MarcoMemory(state_dim=state_dim,
                             action_dim=action_dim,
                             memory_aggr=mem_aggr,
                             k=k,
                             n_memories=n_memories,
                             batch_size=batch_size,
                             pomo_size=pomo_size,
                             repeat_punishment=repeat_punishment,
                             device=device)

    elif memory_type == 'none':
        memory = NoMemory()

    else:
        raise ValueError(f"Memory type {memory_type} is not supported. Use 'operation' or 'marco'.")

    return memory
