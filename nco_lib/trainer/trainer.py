from abc import ABC, abstractmethod
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nco_lib.environment.env import Env
from nco_lib.environment.problem_def import State
from nco_lib.environment.memory import NoMemory



class Trainer(ABC):
    def __init__(self, model: nn.Module, env: Env, optimizer: torch.optim.Optimizer, device: torch.device):
        """
        Trainer parent class with four methods:
            inference: for running the model in inference mode.
            train: for training the model.
            save_checkpoint: for saving the model and optimizer state.
            load_checkpoint: for loading the model and optimizer state.

        :param model: The model to be trained. Type: nn.Module.
        :param env: The environment for the model. Type: Env.
        :param optimizer: The optimizer for the model. Type: torch.optim.Optimizer.
        :param device: The device to run the model on. Type: torch.device.
        """
        self.model = model
        self.model.to(device)
        self.env = env
        self.optimizer = optimizer
        self.device = device

        # Check node and edge dimensions are equal in the model and env.problem
        # first check if the model has an attribute node_in_dim
        if hasattr(self.model, 'node_in_dim'):
            assert self.model.node_in_dim == self.env.problem.node_in_dim + self.env.mem_dim, "Node input dimensions do not match. Model: {}, Problem def: {}".format(self.model.node_in_dim, self.env.problem.node_in_dim + self.env.mem_dim)

        if hasattr(self.model, 'edge_in_dim'):
            assert self.model.edge_in_dim == self.env.problem.edge_in_dim, "Edge input dimensions do not match. Model: {}, Problem def: {}".format(self.model.edge_in_dim, self.env.problem.edge_in_dim)

    @abstractmethod
    def inference(self, **kwargs):
        """
        Abstract method to run the model in inference mode.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, **kwargs):
        """
        Abstract method to train the model.
        """
        raise NotImplementedError

    @abstractmethod
    def _fast_eval(self, **kwargs):
        """
        Abstract method for fast evaluation of the model.
        """
        raise NotImplementedError

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """
        Save the model and optimizer state.
        :param path: The path to save the model. Type: str.
        :param epoch: The epoch number. Type: int.
        """
        if not path.endswith('.pth'):
            path += '.pth'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load the model and optimizer state.
        :param path: The path to load the model. Type: str.

        :return: The epoch number. Type: int.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']


class ConstructiveTrainer(Trainer):
    def __init__(self, model: nn.Module, env: Env, optimizer: torch.optim.Optimizer, device: torch.device):
        """
        Trainer class for constructive problems.
        :param model: The model to be trained. Type: nn.Module.
        :param env: The environment for the model. Type: Env.
        :param optimizer: The optimizer for the model. Type: torch.optim.Optimizer.
        :param device: The device to run the model on. Type: torch.device.
        """
        super().__init__(model, env, optimizer, device)

    @torch.no_grad()
    def inference(self, problem_size: int, batch_size: int, pomo_size: int, deterministic: bool = True,
                  seed: int or None = None, verbose: bool = False) -> (torch.Tensor, dict):
        """
        Run the constructive model in inference mode.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param deterministic: If True, choose the action with the highest probability. Type: bool.
        :param seed: The seed for reproducibility. Type: int or None.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.

        :return: The objective value of the proposed solutions. Type: torch.Tensor.
        :return: The result dictionary. Type: dict.
        """

        # Constructive trainer, therefore, the reward and stopping criteria should be constructive also
        assert self.env.reward.__class__.__name__ == 'ConstructiveReward', "The reward class should be ConstructiveReward"
        assert self.env.stopping_criteria.__class__.__name__ == 'ConstructiveStoppingCriteria', "The stopping criteria should be ConstructiveStoppingCriteria"

        # Start timer and result dictionary
        start_time = time.time()
        result_dict = {
            'problem_size': problem_size,
            'batch_size': batch_size,
            'deterministic': deterministic,
            'seed': seed,

            'final_obj_values': None,
            'final_solutions': None,
            'obj_values': [],
            'elapsed_times': [],
            'logits': [],
        }

        # Set the model to evaluation mode
        self.model.eval()

        if verbose:
            print("\nStart of model inference")

        # Reset the environment
        state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size,
                                                        pomo_size=pomo_size, seed=seed)

        # Main loop
        while not done:
            # Get logits
            logits, aux_node = self.model(state)
            logits = logits + state.mask.reshape(batch_size*pomo_size, logits.size(1), -1)  # mask invalid actions

            logits = logits.reshape(batch_size*pomo_size, -1)  # reshape logits

            # Get actions from the model logits
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1) # softmax to get probabilities
                action = probs.multinomial(1).squeeze(dim=1)  # sample action from the distribution

            # Reshape to (batch_size, pomo_size)
            action = action.reshape(batch_size, pomo_size)

            # Set actions as a Tuple (Selected node/edge, Selected class)
            if self.model.out_dim > 1:
                action = (action // self.model.out_dim, action % self.model.out_dim)
            else:
                action = (action, torch.empty(0))

            # Take a step in the environment
            state, reward, obj_value, done = self.env.step(action)

            # Store the results
            result_dict['obj_values'].append(obj_value.mean().item())
            result_dict['elapsed_times'].append(time.time() - start_time)
            result_dict['logits'].append(logits)

        # Reshape the objective value
        obj_value = obj_value.reshape(batch_size, pomo_size)

        # End-of-inference
        elapsed_time = time.time() - start_time

        # Store the final results
        result_dict['final_obj_values'] = obj_value
        result_dict['final_solutions'] = state.solutions

        if verbose:
            avg_objective_value = obj_value.mean().item()
            # max in dim 1 to get the best solution in each batch
            best_objective_value = obj_value.max(dim=1).values.mean().item()
            print(f"Avg Objective Value: {avg_objective_value:.4f}. Best Objective Value: {best_objective_value:.4f}. Elapsed Time: {elapsed_time:.2f} sec")

        return result_dict

    def train(self, epochs: int, episodes: int, problem_size: int or range, batch_size: int, pomo_size: int,
              eval_problem_size: int, eval_batch_size: int, baseline_type: str = 'mean', max_clip_norm: float = 1.0, eval_freq: int = 1,
              save_freq: int = 1, save_path: str = '', seed: int = 42, verbose: bool = False) -> dict:
        """
        Train the constructive model.
        :param epochs: The number of epochs to train. Type: int.
        :param episodes: The number of episodes to train in each epoch. Type: int.
        :param problem_size: The size of the problem (can be a tuple with the range of sizes). Type: int or range.
        :param batch_size: The size of the batch. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param eval_problem_size: The size of the evaluation problem. Type: int.
        :param eval_batch_size: The size of the evaluation batch. Type: int.
        :param baseline_type: The baseline type to use. Options: 'mean', 'scst', 'pomo' Type: str.
        :param max_clip_norm: The maximum norm for clipping gradients. Type: float.
        :param eval_freq: The frequency (in epochs) of evaluating the model. Type: int.
        :param save_freq: The frequency (in epochs) of saving the model. Type: int.
        :param save_path: The path to save the model. Type: str.
        :param seed: The seed for reproducibility. Type: int.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.

        :return: The result dictionary. Type: dict.
        """
        # Test the problem definition
        test_problem_def(self.model, self.env, problem_size, batch_size, pomo_size)

        # Warnings
        if baseline_type == 'pomo' and pomo_size == 1:
            print("Warning: POMO baseline type is selected, but pomo_size is 1. Changing advantage type to 'mean'.")
            baseline_type = 'mean'

        # Constructive trainer, therefore, the reward and stopping criteria should be constructive also
        assert self.env.reward.__class__.__name__ == 'ConstructiveReward', "The reward class should be ConstructiveReward"
        assert self.env.stopping_criteria.__class__.__name__ == 'ConstructiveStoppingCriteria', "The stopping criteria should be ConstructiveStoppingCriteria"

        # Initialize timer and result dictionary
        start_time = time.time()
        result_dict = {
            'n_epochs': epochs,
            'n_episodes': episodes,
            'problem_size': problem_size,
            'batch_size': batch_size,
            'pomo_size': pomo_size,
            'eval_batch_size': eval_batch_size,
            'seed': seed,

            'loss_values': [],
            'obj_values': [],
            'elapsed_times': [],

            'eval_obj_values': [],
            'eval_elapsed_times': []
        }

        # Generate the evaluation batch
        eval_state, _, eval_obj_value, _ = self.env.reset(problem_size=eval_problem_size, batch_size=eval_batch_size,
                                                          pomo_size=pomo_size, seed=seed)

        if verbose:
            print("\nStart of model training")

        # Initialize the batch range
        batch_pomo_range = torch.arange(batch_size*pomo_size)

        # Training loop
        for epoch in range(epochs):
            # Set the model to training mode
            self.model.train()

            for episode in range(episodes):
                # Reset the environment
                state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size,
                                                                pomo_size=pomo_size)

                # Initialize the probability list
                prob_list = torch.zeros(size=(batch_size*pomo_size, 0), device=self.device)

                # Main inference loop
                self.optimizer.zero_grad()
                while not done:
                    # Get logits (shape: batch_size x (num_nodes or num_edges) x output_dim)
                    logits, aux_node = self.model(state)

                    # Mask invalid actions
                    logits = logits + state.mask.reshape(batch_size*pomo_size, logits.size(1), -1)

                    # Reshape logits to perform softmax (for multi-class)
                    logits = logits.reshape(batch_size*pomo_size, -1)  # reshape logits

                    # Get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # Sample action from the distribution
                    action = probs.multinomial(1).squeeze(dim=1)

                    # Get the probability of the selected action and append it to the list
                    prob = probs[batch_pomo_range, action]
                    prob_list = torch.cat((prob_list, prob[:, None]), dim=1)

                    # Reshape to (batch_size, pomo_size)
                    action = action.reshape(batch_size, pomo_size)

                    # Set actions as a Tuple (Selected node/edge, Selected class)
                    if self.model.out_dim > 1:
                        action = (action // self.model.out_dim, action % self.model.out_dim)
                    else:
                        action = (action, torch.empty(0))

                    # Take a step in the environment
                    state, reward, obj_value, done = self.env.step(action)

                # Once an episode is done, calculate the loss
                log_prob = prob_list.log().sum(dim=1)
                if baseline_type == 'mean':
                    # Use the average reward as the baseline, and subtract it from the reward to get the advantage
                    advantage = reward - reward.mean()
                elif baseline_type == 'pomo':
                    # Use POMO, subtract the mean reward of each pomo from the reward
                    reward = reward.reshape(batch_size, pomo_size)  # reshape the reward to (batch_size, pomo_size)
                    advantage = reward - reward.float().mean(dim=1, keepdims=True)
                    advantage = advantage.reshape(-1)  # reshape the advantage to (batch_size*pomo_size)
                else:
                    raise NotImplementedError

                loss = -advantage * log_prob  # Minus Sign: To Increase REWARD

                # Calculate the mean loss and back-propagate
                loss_mean = loss.mean()
                loss_mean.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                # Update the model
                self.optimizer.step()

                # Append results
                elapsed_time = time.time() - start_time
                result_dict['loss_values'].append(loss_mean.item())
                result_dict['obj_values'].append(obj_value.mean().item())
                result_dict['elapsed_times'].append(elapsed_time)

                if verbose:
                    print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss_mean.item():.4f}, Obj value: {obj_value.mean().item():.3f}, Elapsed Time: {elapsed_time:.2f} sec")

            # Evaluate the model
            if (epoch + 1) % eval_freq == 0:
                elapsed_time = time.time() - start_time
                obj_values = self._fast_eval(eval_state, eval_obj_value)
                result_dict['eval_obj_values'].append(obj_values.mean().item())
                result_dict['eval_elapsed_times'].append(elapsed_time)
                if verbose:
                    print(f"Epoch {epoch + 1}, Eval Obj value: {obj_values.mean().item():.3f}, Elapsed Time: {elapsed_time:.2f} sec")

            # Save the model every save_freq epochs
            if save_path and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_path, epoch)

        # End of training loop - Save the final model
        if save_path:
            self.save_checkpoint(save_path, epochs)

        if verbose:
            print(f"End of model training. Model saved. Total Elapsed Time: {time.time() - start_time:.2f} sec")

        return result_dict

    @torch.no_grad()
    def _fast_eval(self, eval_state: State, eval_obj_value: torch.Tensor) -> torch.Tensor:
        """
        Fast evaluation of the model.
        :param eval_state: The evaluation state. Type: State
        :param eval_obj_value: The evaluation state's initial objective value. Type: torch.Tensor
        """
        # Set the model to evaluation
        self.model.eval()

        # Reset the environment
        state, reward, obj_value, done = self.env.set(eval_state, eval_obj_value)

        # Main loop
        while not done:
            # Get logits
            logits, aux_node = self.model(state)
            logits = logits + state.mask.reshape(state.batch_size*state.pomo_size, logits.size(1), -1)  # mask invalid actions
            logits = logits.reshape(state.batch_size*state.pomo_size, -1)  # reshape logits

            # Get actions from the model logits
            action = logits.argmax(dim=-1)

            # Reshape to (batch_size, pomo_size)
            action = action.reshape(state.batch_size, state.pomo_size)

            # Set actions as a Tuple (Selected node/edge, Selected class)
            if self.model.out_dim > 1:
                action = (action // self.model.out_dim, action % self.model.out_dim)
            else:
                action = (action, torch.empty(0))

            # Take a step in the environment
            state, reward, obj_value, done = self.env.step(action)

        return obj_value


class ImprovementTrainer(Trainer):
    def __init__(self, model: nn.Module, env: Env, optimizer: torch.optim.Optimizer, device: torch.device):
        """
        Trainer class for improvement problems.
        """
        super().__init__(model, env, optimizer, device)

    @torch.no_grad()
    def inference(self, problem_size: int, batch_size: int, pomo_size: int, deterministic: bool = True,
                  seed: int or None = None, verbose: bool = False) -> (torch.Tensor, dict):
        """
        Run the improvement model in inference mode.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param deterministic: If True, choose the action with the highest probability. Type: bool.
        :param seed: The seed for reproducibility. Type: int or None.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.

        :return: The objective value of the proposed solutions. Type: torch.Tensor.
        :return: The result dictionary. Type: dict.
        """

        # Improvement trainer, therefore, the reward and stopping criteria should be improvement also
        assert self.env.reward.__class__.__name__ == 'ImprovementReward', "The reward class should be ImprovementReward"
        assert self.env.stopping_criteria.__class__.__name__ == 'ImprovementStoppingCriteria', "The stopping criteria should be ImprovementStoppingCriteria"

        # Start timer and result dictionary
        start_time = time.time()
        result_dict = {
            'problem_size': problem_size,
            'batch_size': batch_size,
            'deterministic': deterministic,
            'seed': seed,

            'final_obj_values': None,
            'final_solutions': None,
            'avg_obj_values': [],
            'best_obj_values': [],
            'overall_best_obj_values': [],
            'elapsed_times': [],
            'logits': [],
        }

        # Set the model to evaluation mode
        self.model.eval()

        if verbose:
            print("\nStart of model inference")

        # Reset the environment
        state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size,
                                                        pomo_size=pomo_size, seed=seed)

        # Main inference loop
        step = 0
        best_obj_values = obj_value.max(dim=1).values
        while not done:
            step += 1

            # Get logits
            logits, aux_node = self.model(state)

            # Mask invalid actions
            logits = logits + state.mask.reshape(batch_size*pomo_size, logits.size(1), -1)  # mask invalid actions

            logits = logits.reshape(batch_size*pomo_size, -1)  # reshape logits

            # Get actions from the model logits
            if deterministic:
                action = logits.argmax(dim=-1)  # choose the action with the highest probability
            else:
                probs = F.softmax(logits, dim=-1)  # softmax to get probabilities
                action = probs.multinomial(1).squeeze(dim=1)  # sample action from the distribution

            # Reshape to (batch_size, pomo_size)
            action = action.reshape(batch_size, pomo_size)

            # Set actions as a Tuple (Selected node/edge, Selected class)
            if self.model.out_dim > 1:
                action = (action // self.model.out_dim, action % self.model.out_dim)
            else:
                action = (action, torch.empty(0))

            # Take a step in the environment
            state, reward, obj_value, done = self.env.step(action)

            # Update best objective values
            best_obj_values = torch.max(best_obj_values, obj_value.max(dim=1).values)

            # Store the results
            result_dict['avg_obj_values'].append(obj_value.mean().item())
            result_dict['best_obj_values'].append(obj_value.max(dim=1).values.mean().item())
            result_dict['overall_best_obj_values'].append(best_obj_values.mean().item())
            result_dict['elapsed_times'].append(time.time() - start_time)
            result_dict['logits'].append(logits)

            if verbose:
                print(f"Step {step}. Obj value: {obj_value.mean().item()}")

        # End of inference
        elapsed_time = time.time() - start_time

        # Store the final results
        result_dict['final_obj_values'] = obj_value
        result_dict['final_solutions'] = state.solutions

        if verbose:
            print(f"Best Objective Value: {best_obj_values.mean().item():.4f}. Elapsed Time: {elapsed_time:.2f} sec")

        return obj_value, result_dict

    def train(self, epochs: int, episodes: int, problem_size: int, batch_size: int, pomo_size: int,
              eval_problem_size: int, eval_batch_size: int, baseline_type: str = 'mean', update_freq: int = 10,
              gamma: float = 0.99, max_clip_norm: float = 1.0, eval_freq: int = 1, save_freq: int = 1,
              save_path: str = '', seed: int = 42, verbose: bool = False) -> dict:
        """
        Train the improvement model.
        :param epochs: The number of epochs to train. Type: int.
        :param episodes: The number of episodes to train in each epoch. Type: int.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param eval_problem_size: The size of the evaluation problem. Type: int.
        :param eval_batch_size: The size of the evaluation batch. Type: int.
        :param baseline_type: The baseline type to use. Options: 'mean', 'scst', 'pomo' Type: str.
        :param update_freq: The frequency of updating the model. Type: int.
        :param gamma: The discount factor to determine the importance of future rewards. Type: float.
        :param max_clip_norm: The maximum norm for clipping gradients. Type: float.
        :param eval_freq: The frequency (in epochs) of evaluating the model. Type: int.
        :param save_freq: The frequency (in epochs) of saving the model. Type: int.
        :param save_path: The path to save the model. Type: str.
        :param seed: The seed for reproducibility. Type: int.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.
        """
        # Test the problem definition
        test_problem_def(self.model, self.env, problem_size, batch_size, pomo_size)

        # Start timer and result dictionary
        start_time = time.time()
        result_dict = {
            'n_epochs': epochs,
            'n_episodes': episodes,
            'problem_size': problem_size,
            'batch_size': batch_size,
            'pomo_size': pomo_size,
            'eval_batch_size': eval_batch_size,

            'loss_values': [],
            'obj_values': [],
            'elapsed_times': [],

            'eval_obj_values': [],
            'eval_elapsed_times': []
        }

        # Generate the evaluation batch
        eval_state, _, eval_obj_value, _ = self.env.reset(problem_size=eval_problem_size, batch_size=eval_batch_size,
                                                          pomo_size=pomo_size, seed=seed)

        if verbose:
            print("\nStart of model training")

        # Initialize the batch range
        batch_pomo_range = torch.arange(batch_size*pomo_size)

        # Training loop
        for epoch in range(epochs):
            # Set the model to training mode
            self.model.train()
            avg_loss = 0
            avg_obj_value = 0
            update_count = 0
            for episode in range(episodes):
                # Reset the environment
                state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size,
                                                                pomo_size=pomo_size)

                # Initialize the best objective value found so far
                best_obj_value = obj_value.clone()

                # Episode loop
                episode_steps = 0
                while not done:
                    # Update loop
                    log_probs = []
                    rewards = []
                    for step in range(update_freq):
                        episode_steps += 1

                        # Get logits
                        logits, aux_node = self.model(state)

                        # Mask invalid actions
                        logits = logits + state.mask.reshape(batch_size * pomo_size, logits.size(1), -1)

                        # Reshape logits to perform softmax (for multi-class)
                        logits = logits.reshape(batch_size * pomo_size, -1)  # reshape logits

                        # Softmax to get probabilities
                        probs = torch.nn.functional.softmax(logits, dim=-1)

                        # Sample action from the distribution
                        action = probs.multinomial(1).squeeze(dim=1)
                        prob = probs[batch_pomo_range, action]

                        # Reshape to (batch_size, pomo_size)
                        action = action.reshape(batch_size, pomo_size)

                        # Set actions as a Tuple (Selected node/edge, Selected class)
                        if self.model.out_dim > 1:
                            action = (action // self.model.out_dim, action % self.model.out_dim)
                        else:
                            action = (action, torch.zeros_like(action))

                        # Take a step in the environment
                        state, reward, obj_value, done = self.env.step(action)

                        # Update the best objective value found so far
                        best_obj_value = torch.max(best_obj_value, obj_value)

                        # Append log probabilities and rewards
                        log_probs.append(prob.log())
                        rewards.append(reward.reshape(batch_size*pomo_size))

                        if done:
                            break

                    # Update frequency reached -> Calculate discounting rewards
                    t_steps = torch.arange(len(rewards))
                    discounts = gamma ** t_steps
                    r = [r_i * d_i for r_i, d_i in zip(rewards, discounts)]
                    r = r[::-1]  # reverse the list
                    b = torch.cumsum(torch.stack(r), dim=0)  # calculate the cumulative sum
                    c = [b[k, :] for k in reversed(range(b.shape[0]))]  # reverse the list
                    R = [c_i / d_i for c_i, d_i in zip(c, discounts)]  # divide by discount factor

                    # Calculate the loss and backpropagate
                    self.optimizer.zero_grad()
                    policy_loss = []
                    for log_prob_i, r_i in zip(log_probs, R):
                        policy_loss.append(-log_prob_i * r_i)
                    loss = torch.cat(policy_loss).mean()
                    loss.backward()

                    # Add the loss and objective value
                    avg_loss += loss.item()
                    avg_obj_value += best_obj_value.mean().item()
                    update_count += 1

                    # Clip grad norms
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                    # Update the model
                    self.optimizer.step()

                if verbose:
                    print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {avg_loss / update_count:.4f}, "
                          f"Reward: {reward.mean().item():.3f}, Obj value: {avg_obj_value / update_count:.3f}, "
                          f"Best Obj value: {best_obj_value.mean().item():.3f}, "
                          f"Steps: {episode_steps}, Elapsed Time: {time.time() - start_time:.2f} sec")

                # Append results
                result_dict['loss_values'].append(avg_loss / update_count)
                result_dict['obj_values'].append(avg_obj_value / update_count)
                result_dict['elapsed_times'].append(time.time() - start_time)

            # Evaluate the model
            if (epoch + 1) % eval_freq == 0:
                elapsed_time = time.time() - start_time
                obj_values = self._fast_eval(eval_state, eval_obj_value)
                result_dict['eval_obj_values'].append(obj_values.mean().item())
                result_dict['eval_elapsed_times'].append(elapsed_time)
                if verbose:
                    print(f"Epoch {epoch + 1}, Eval Obj value: {obj_values.mean().item():.3f}, Elapsed Time: {elapsed_time:.2f} sec")

            if save_path and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_path, epoch)

        # End of training loop - Save the final model
        if save_path:
            self.save_checkpoint(save_path, epochs)

        if verbose:
            print(f"End of model training. Model saved. Total Elapsed Time: {time.time() - start_time:.2f} sec")

        return result_dict

    @torch.no_grad()
    def _fast_eval(self, eval_state: State, eval_obj_value: torch.Tensor) -> torch.Tensor:
        """
        Fast evaluation of the model.
        :param eval_state: The evaluation state. Type: State
        :param eval_obj_value: The evaluation state's initial objective value. Type: torch.Tensor
        """
        # Set the model to evaluation
        self.model.eval()

        # Reset the environment
        state, reward, obj_value, done = self.env.set(eval_state, eval_obj_value)

        # Main inference loop
        step = 0
        best_obj_values = obj_value.max(dim=1).values
        while not done:
            step += 1

            # Get logits
            logits, aux_node = self.model(state)

            # Mask invalid actions
            logits = logits + state.mask.reshape(state.batch_size * state.pomo_size, logits.size(1), -1)  # mask invalid actions
            logits = logits.reshape(state.batch_size * state.pomo_size, -1)  # reshape logits

            # Get actions from the model logits
            action = logits.argmax(dim=-1)  # choose the action with the highest probability

            # Reshape to (batch_size, pomo_size)
            action = action.reshape(state.batch_size, state.pomo_size)

            # Set actions as a Tuple (Selected node/edge, Selected class)
            if self.model.out_dim > 1:
                action = (action // self.model.out_dim, action % self.model.out_dim)
            else:
                action = (action, torch.zeros_like(action))

            # Take a step in the environment
            state, reward, obj_value, done = self.env.step(action)

            # Update best objective values
            best_obj_values = torch.max(best_obj_values, obj_value.max(dim=1).values)

        return best_obj_values



def test_problem_def(model: nn.Module, env: Env, problem_size: int, batch_size: int, pomo_size: int):
    """
    Test the problem defined by the user.
        - The state.node_features should be of shape (batch_size, pomo_size, num_nodes, num_features).
        - The state.edge_features should be of shape (batch_size, pomo_size, num_nodes, num_nodes, num_features).
        - The state.adj_matrix should be a tensor of shape: (batch_size, num_nodes, num_nodes).
        - The state.mask should be a tensor of shape: (batch_size, pomo_size, num_nodes, 1).

    Test the model forward pass.
        - The logits should be of shape: (batch_size*pomo_size, num_nodes, num_classes).
        - The action should be of shape: (batch_size*pomo_size, 1).

    TODO: add tests for the model type: input/output nodes/edges

    """

    def test_state(state: State):
        # node_features has three dimensions: batch_size, pomo_size, num_nodes, num_features
        assert state.node_features is not None
        assert state.node_features.dim() == 4
        num_node_features = state.node_features.shape[-1]
        assert state.node_features.shape == (batch_size, pomo_size, problem_size, num_node_features)

        # edge_features has four dimensions: batch_size, pomo_size, num_nodes, num_nodes, num_features
        if state.edge_features is not None:
            assert state.edge_features.dim() == 5
            num_edge_features = state.edge_features.shape[-1]
            assert state.edge_features.shape == (batch_size, pomo_size, problem_size, problem_size, num_edge_features)

        # adj_matrix has three dimensions: batch_size, num_nodes, num_nodes
        if state.adj_matrix is not None:
            assert state.adj_matrix.dim() == 3
            assert state.adj_matrix.shape == (batch_size, problem_size, problem_size)

        # mask has three dimensions: batch_size, pomo_size, num_nodes, 1
        if state.mask is not None:
            assert state.mask.dim() == 4
            assert state.mask.shape == (batch_size, pomo_size, problem_size, 1) or state.mask.shape == (batch_size, pomo_size, problem_size**2, 1)


    cur_state, reward, obj_value, done = env.reset(problem_size=problem_size, batch_size=batch_size,
                                                   pomo_size=pomo_size)

    test_state(cur_state)

    # STEP
    logits, aux_node = model(cur_state)
    logits = logits + cur_state.mask.reshape(batch_size * pomo_size, logits.size(1), -1)  # Mask invalid actions
    logits = logits.reshape(batch_size * pomo_size, -1)  # reshape logits
    probs = torch.nn.functional.softmax(logits, dim=-1)  # Get probabilities
    action = probs.multinomial(1).squeeze(dim=1)  # Sample action from the distribution

    # Reshape to (batch_size, pomo_size)
    action = action.reshape(batch_size, pomo_size)

    # Set actions as a Tuple (Selected node/edge, Selected class)
    if model.out_dim > 1:
        action = (action // model.out_dim, action % model.out_dim)
    else:
        action = (action, torch.empty(0))
    cur_state, reward, obj_value, done = env.step(action)

    # Test the model forward pass
    test_state(cur_state)
