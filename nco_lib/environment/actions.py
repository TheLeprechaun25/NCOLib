from abc import ABC, abstractmethod
import torch


class Action(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, solutions, actions, batch_range):
        pass



def bit_flip(solutions, actions):
    """
    Apply the action to the state.
    """
    batch_size = solutions.shape[0]
    batch_range = torch.arange(batch_size, device=solutions.device)
    solutions[batch_range, actions] = 1 - solutions[batch_range, actions]
    return solutions


def swap(solutions, actions):
    """
    Apply the action to the state.
    """
    batch_size, problem_size = solutions.shape
    idx1 = actions // problem_size
    idx2 = actions % problem_size

    # Create a copy to avoid modifying the original tensor
    new_solutions = solutions.clone()
    for i in range(batch_size):
        # Swap positions for each row
        temp = new_solutions[i, idx1[i]].clone()
        new_solutions[i, idx1[i]] = new_solutions[i, idx2[i]]
        new_solutions[i, idx2[i]] = temp
    return new_solutions


def insert(solutions, actions):
    """
    Apply the action to the state.
    """
    batch_size, problem_size = solutions.shape
    idx1 = actions // problem_size
    idx2 = actions % problem_size
    idx1 = idx1.unsqueeze(1)
    idx2 = idx2.unsqueeze(1)
    new_solutions = solutions.clone()

    # fix connection for first node
    argsort = solutions.argsort()

    pre_first = argsort.gather(1, idx1)
    post_first = solutions.gather(1, idx1)

    new_solutions.scatter_(1, pre_first, post_first)

    # fix connection for second node
    post_second = new_solutions.gather(1, idx2)

    new_solutions.scatter_(1, idx2, idx1)
    new_solutions.scatter_(1, idx1, post_second)

    return new_solutions


def two_opt(solutions, actions):
    """
    Apply the action to the state.
    """
    batch_size, problem_size = solutions.shape
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

    return new_solutions
