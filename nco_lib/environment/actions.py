from abc import ABC, abstractmethod
import torch
import numpy as np


class Action(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, solutions, actions, batch_range):
        pass


def bit_flip(solutions, actions):
    """
    Apply the action to the state.
    :param solutions: The current state of the solutions. Shape: (batch_size, pomo_size, problem_size).
    :param actions: The actions to apply to the state. Shape: (batch_size, pomo_size).
    """
    batch_size, pomo_size = solutions.shape[:2]
    batch_range = torch.arange(batch_size, device=solutions.device)[:, None].expand(batch_size, pomo_size)
    pomo_range = torch.arange(pomo_size, device=solutions.device)[None, :].expand(batch_size, pomo_size)
    solutions[batch_range, pomo_range, actions] = 1 - solutions[batch_range, pomo_range, actions]
    return solutions


def swap(solutions, actions):
    """
    Apply the action to the state.
    Action refers to an edge to swap.
    """
    batch_size, pomo_size, problem_size = solutions.shape
    solutions = solutions.view(batch_size * pomo_size, problem_size)
    actions = actions.view(batch_size * pomo_size)

    idx1 = actions // problem_size
    idx2 = actions % problem_size

    # Create a copy to avoid modifying the original tensor
    new_solutions = solutions.clone()
    for i in range(batch_size*pomo_size):
        # Swap positions for each row
        temp = new_solutions[i, idx1[i]].clone()
        new_solutions[i, idx1[i]] = new_solutions[i, idx2[i]]
        new_solutions[i, idx2[i]] = temp
    return new_solutions.view(batch_size, pomo_size, problem_size)


def insert(solutions, actions):
    """
    Apply the action to the state.
    """
    batch_size, pomo_size, problem_size = solutions.shape
    solutions = solutions.view(batch_size * pomo_size, problem_size)
    actions = actions.view(batch_size * pomo_size)

    node1 = actions // problem_size
    node2 = actions % problem_size
    new_solution = solutions.clone()
    for b in range(batch_size*pomo_size):
        sol = solutions[b].cpu().numpy()
        # where is the node1 in the solution
        idx_node_1 = node1[b].item() #np.where(sol == node1[b].item())[0][0]
        idx_node_2 = node2[b].item() #np.where(sol == node2[b].item())[0][0]

        item = sol[idx_node_1]
        new_order = np.delete(sol, idx_node_1)
        new_order = np.insert(new_order, idx_node_2, item)
        new_solution[b] = torch.tensor(new_order, device=solutions.device)

    return new_solution.view(batch_size, pomo_size, problem_size)

    node1 = actions // problem_size
    node2 = actions % problem_size
    node1 = node1.unsqueeze(1)
    node2 = node2.unsqueeze(1)
    new_solutions = solutions.clone()

    # Fix connection for first node
    argsort = solutions.argsort()

    pre_first = argsort.gather(1, node1)
    post_first = solutions.gather(1, node1)

    new_solutions.scatter_(1, pre_first, post_first)

    # fix connection for second node
    post_second = new_solutions.gather(1, node2)

    new_solutions.scatter_(1, node2, node1)
    new_solutions.scatter_(1, node1, post_second)

    return new_solutions.view(batch_size, pomo_size, problem_size)


def two_opt(solutions, actions):
    """
    Apply the action to the state.
    """
    batch_size, pomo_size, problem_size = solutions.shape
    solutions = solutions.view(batch_size * pomo_size, problem_size)
    actions = actions.view(batch_size * pomo_size)

    idx1 = actions // problem_size
    idx2 = actions % problem_size
    idx1 = idx1.unsqueeze(1)
    idx2 = idx2.unsqueeze(1)
    new_solutions = solutions.clone()

    # fix connection for first node
    argsort = solutions.argsort()
    pre_first = argsort.gather(1, idx1)
    pre_first = torch.where(pre_first != idx2, pre_first, idx1)
    new_solutions.scatter_(1, pre_first, idx2)

    # fix connection for second node
    post_second = solutions.gather(1, idx2)
    post_second = torch.where(post_second != idx1, post_second, idx2)
    new_solutions.scatter_(1, idx1, post_second)

    # reverse loop:
    cur = idx1
    for j in range(problem_size):
        cur_next = solutions.gather(1, cur)
        new_solutions.scatter_(1, cur_next, torch.where(cur != idx2, cur, new_solutions.gather(1, cur_next)))
        cur = torch.where(cur != idx2, cur_next, cur)

    return new_solutions.view(batch_size, pomo_size, problem_size)
