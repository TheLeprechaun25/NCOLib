from abc import ABC, abstractmethod
import time
from pathlib import Path
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from nco_lib.environment.env import Env
from nco_lib.environment.problem_def import State



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

    def save_checkpoint(self, cur_run_name: str, epoch: int) -> str:
        """
        Save the model and optimizer state.
        :param cur_run_name: The path to save the model. Type: str.
        :param epoch: The epoch number. Type: int.
        """
        run_dir = Path(f"../runs/{cur_run_name}/")
        run_dir.mkdir(parents=True, exist_ok=True)

        path = run_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        return str(path)

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
    def inference(self, problem_size: int or None = None, batch_size: int or None = None, pomo_size: int = 1, deterministic: bool = True,
                  use_custom_instances: bool = False, seed: int or None = None, verbose: bool = False) -> (torch.Tensor, dict):
        """
        Run the constructive model in inference mode.
        :param problem_size: The size of the problem. Type: int or None.
        :param batch_size: The size of the batch. Type: int or None.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param deterministic: If True, choose the action with the highest probability. Type: bool.
        :param use_custom_instances: If True, a custom instance will be loaded. Type: bool.
        :param seed: The seed for reproducibility. Type: int or None.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.

        :return: The objective value of the proposed solutions. Type: torch.Tensor.
        :return: The result dictionary. Type: dict.
        """

        # Constructive trainer, therefore, the reward and stopping criteria should be constructive also
        assert self.env.reward.__class__.__name__ == 'ConstructiveReward', "The reward class should be ConstructiveReward"
        assert self.env.stopping_criteria.__class__.__name__ == 'ConstructiveStoppingCriteria', "The stopping criteria should be ConstructiveStoppingCriteria"

        if not use_custom_instances:
            assert problem_size is not None, "The problem size should be provided"
            assert batch_size is not None, "The batch size should be provided"

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
                                                        pomo_size=pomo_size, use_custom_instances=use_custom_instances, seed=seed)

        # Check if batch_size, pomo_size or problem_size have been modified while using custom instances
        batch_size = state.batch_size
        pomo_size = state.pomo_size
        problem_size = state.problem_size

        # Main loop
        while not done:
            # Get logits
            logits = self.model(state)
            if state.mask is not None:
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
              eval_problem_size: int, eval_batch_size: int, learn_algo: str = 'reinforce', baseline_type: str = 'mean',
              max_clip_norm: float = 1.0, eval_freq: int = 1, save_freq: int = 1, save_path_name: str = '',
              seed: int = 42, verbose: bool = False) -> dict:
        """
        Train the constructive model.
        :param epochs: The number of epochs to train. Type: int.
        :param episodes: The number of episodes to train in each epoch. Type: int.
        :param problem_size: The size of the problem (can be a tuple with the range of sizes). Type: int or range.
        :param batch_size: The size of the batch. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param eval_problem_size: The size of the evaluation problem. Type: int.
        :param eval_batch_size: The size of the evaluation batch. Type: int.
        :param learn_algo: The Learning Algorithm. Options: 'reinforce', 'ppo'. Type: str.
        :param baseline_type: The baseline type to use. Options: 'mean', 'scst', 'pomo' Type: str.
        :param max_clip_norm: The maximum norm for clipping gradients. Type: float.
        :param eval_freq: The frequency (in epochs) of evaluating the model. Type: int.
        :param save_freq: The frequency (in epochs) of saving the model. Type: int.
        :param save_path_name: The name to save the model. Type: str.
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

        # Initialize the run name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cur_run_name = f"NC_train_{save_path_name}_{timestamp}"

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

        # Training loop
        for epoch in range(epochs):
            # Set the model to training mode
            self.model.train()
            if learn_algo == 'reinforce':
                result_dict = self._train_reinforce(episodes, problem_size, batch_size, pomo_size, baseline_type,
                                                    max_clip_norm, start_time, result_dict, epoch, verbose)
            elif learn_algo == 'ppo':
                result_dict = self._train_ppo(episodes, problem_size, batch_size, pomo_size, baseline_type,
                                              max_clip_norm, start_time, result_dict, epoch, verbose)
            else:
                raise NotImplementedError

            # Evaluate the model
            if (eval_freq>0) and (epoch + 1) % eval_freq == 0:
                elapsed_time = time.time() - start_time
                obj_values = self._fast_eval(eval_state, eval_obj_value)
                result_dict['eval_obj_values'].append(obj_values.mean().item())
                result_dict['eval_elapsed_times'].append(elapsed_time)
                if verbose:
                    print(f"Epoch {epoch + 1}, Eval Obj value: {obj_values.mean().item():.3f}, Elapsed Time: {elapsed_time:.2f} sec")

            # Save the model every save_freq epochs
            if (save_freq>0) and ((epoch + 1) % save_freq == 0 or (epoch + 1 == epochs)):
                saved_path = self.save_checkpoint(cur_run_name, epoch+1)
                if verbose:
                    print(f"Saving checkpoint in {saved_path}.")

        # End of training loop
        if verbose:
            print(f"End of model training. Total Elapsed Time: {time.time() - start_time:.2f} sec")

        return result_dict

    def _train_reinforce(self, episodes, problem_size, batch_size, pomo_size, baseline_type, max_clip_norm, start_time,
                         result_dict, epoch, verbose):
        # Initialize the range
        batch_pomo_range = torch.arange(batch_size*pomo_size)

        # train loop
        running_obj_avg = 0
        for episode in range(episodes):
            # Reset the environment
            state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size,
                                                            pomo_size=pomo_size)

            # Initialize the probability list
            prob_list = torch.zeros(size=(batch_size * pomo_size, 0), device=self.device)

            # Main inference loop
            while not done:
                # Get logits (shape: batch_size x (num_nodes or num_edges) x output_dim)
                logits = self.model(state)

                # Mask invalid actions
                if state.mask is not None:
                    logits = logits + state.mask.reshape(batch_size * pomo_size, logits.size(1), -1)

                # Reshape logits to perform softmax (for multi-class)
                logits = logits.reshape(batch_size * pomo_size, -1)  # reshape logits

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
                # Use the average reward as the baseline, and subtract it from the reward to get the advantage:
                # advantage = reward - reward.mean()

                # Leave one out
                sum_r = reward.sum(dim=0, keepdim=True)
                baselines_loo = (sum_r - reward) / (reward.shape[0] - 1)
                advantage = reward - baselines_loo

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
            self.optimizer.zero_grad()
            loss_mean.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm) # Clip gradients
            self.optimizer.step() # Update the model

            # Append results
            elapsed_time = time.time() - start_time
            obj_mean = obj_value.mean().item()
            result_dict['loss_values'].append(loss_mean.item())
            result_dict['obj_values'].append(obj_mean)
            result_dict['elapsed_times'].append(elapsed_time)

            if verbose:
                running_obj_avg += obj_mean
                print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss_mean.item():.4f}, Obj value: {obj_mean:.3f} (running avg {running_obj_avg/(episode+1):.3f}), Elapsed Time: {elapsed_time:.2f} sec")
        return result_dict

    def _train_ppo(self, episodes, problem_size, batch_size, pomo_size, baseline_type, max_clip_norm, start_time,
                   result_dict, epoch, verbose):
        # TODO
        raise NotImplementedError


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
            logits = self.model(state)
            if state.mask is not None:
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


class HeatmapTrainer(Trainer):
    def __init__(self, model: nn.Module, env: Env, optimizer: torch.optim.Optimizer, device: torch.device):
        """
        Trainer class for heatmap-based problems.
        :param model: The model to be trained. Type: nn.Module.
        :param env: The environment for the model. Type: Env.
        :param optimizer: The optimizer for the model. Type: torch.optim.Optimizer.
        :param device: The device to run the model on. Type: torch.device.
        """
        super().__init__(model, env, optimizer, device)

    @torch.no_grad()
    def inference(self, problem_size: int or None = None, batch_size: int or None = None, n_rollouts: int = 1, pomo_size: int = 1,
                  deterministic: bool = True, use_custom_instances: bool = False, seed: int or None = None, verbose: bool = False) -> (torch.Tensor, dict):
        """
        Run the heatmap model in inference mode.
        :param problem_size: The size of the problem. Type: int or None.
        :param batch_size: The size of the batch. Type: int or None.
        :param n_rollouts: Number of rollouts (solutions to decode). Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param deterministic: If True, choose the action with the highest probability. Type: bool.
        :param use_custom_instances: If True, custom instances will be used. Type: bool.
        :param seed: The seed for reproducibility. Type: int or None.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.

        :return: The objective value of the proposed solutions. Type: torch.Tensor.
        :return: The result dictionary. Type: dict.
        """

        # Constructive trainer, therefore, the reward and stopping criteria should be constructive also
        assert self.env.reward.__class__.__name__ == 'HeatmapReward', "The reward class should be HeatmapReward"

        if not use_custom_instances:
            assert problem_size is not None, "The problem size should be provided"
            assert batch_size is not None, "The batch size should be provided"

        if pomo_size != 1:
            print("Warning: POMO size should be 1 for heatmap, use n_rollouts instead. Setting pomo_size to 1.")
            pomo_size = 1

        # Start timer and result dictionary
        start_time = time.time()
        result_dict = {
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
                                                        pomo_size=pomo_size, use_custom_instances=use_custom_instances,
                                                        seed=seed)

        # Check if batch_size, pomo_size or problem_size have been modified while using custom instances
        batch_size = state.batch_size
        pomo_size = state.pomo_size
        problem_size = state.problem_size

        # Get logits
        logits = self.model(state)
        if state.mask is not None:
            logits = logits + state.mask.reshape(state.batch_size, logits.size(1), -1)  # mask invalid actions

        # Take a step in the environment
        reward, _, obj_value = self.env.heatmap_decode(logits, n_rollouts=n_rollouts, deterministic=deterministic)

        # Store the results
        result_dict['obj_values'].append(obj_value.mean().item())
        result_dict['elapsed_times'].append(time.time() - start_time)
        result_dict['logits'].append(logits)

        # Reshape the objective value
        obj_value = obj_value.reshape(state.batch_size, n_rollouts)

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

    def train(self, epochs: int, episodes: int, problem_size: int or range, batch_size: int, n_rollouts: int,
              eval_problem_size: int, eval_batch_size: int, pomo_size: int = 1, learn_algo: str = 'reinforce',
              max_clip_norm: float = 1.0, eval_freq: int = 1, save_freq: int = 1, save_path_name: str = '',
              seed: int = 42, verbose: bool = False) -> dict:
        """
        Train the heatmap model.
        :param epochs: The number of epochs to train. Type: int.
        :param episodes: The number of episodes to train in each epoch. Type: int.
        :param problem_size: The size of the problem (can be a tuple with the range of sizes). Type: int or range.
        :param batch_size: The size of the batch. Type: int.
        :param n_rollouts: Number of rollouts to perform. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param eval_problem_size: The size of the evaluation problem. Type: int.
        :param eval_batch_size: The size of the evaluation batch. Type: int.
        :param learn_algo: The Learning Algorithm. Options: 'reinforce', 'ppo'. Type: str.
        :param max_clip_norm: The maximum norm for clipping gradients. Type: float.
        :param eval_freq: The frequency (in epochs) of evaluating the model. Type: int.
        :param save_freq: The frequency (in epochs) of saving the model. Type: int.
        :param save_path_name: The name to save the model. Type: str.
        :param seed: The seed for reproducibility. Type: int.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.

        :return: The result dictionary. Type: dict.
        """
        # TODO: Test the problem definition for heatmap trainer
        #test_problem_def(self.model, self.env, problem_size, batch_size, pomo_size)

        # Warnings
        if pomo_size != 1:
            print("Warning: POMO size should be 1 for heatmap, use n_rollouts instead. Setting pomo_size to 1.")
            pomo_size = 1

        # Heatmap trainer, therefore, the reward and stopping criteria should be constructive also
        assert self.env.reward.__class__.__name__ == 'HeatmapReward', "The reward class should be HeatmapReward"
        assert self.env.stopping_criteria.__class__.__name__ == 'HeatmapStoppingCriteria', "The stopping criteria should be HeatmapStoppingCriteria"

        # Initialize the run name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cur_run_name = f"NC_train_{save_path_name}_{timestamp}"

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

        # Training loop
        for epoch in range(epochs):
            # Set the model to training mode
            self.model.train()
            if learn_algo == 'reinforce':
                result_dict = self._train_reinforce(episodes, problem_size, batch_size, n_rollouts, max_clip_norm, start_time,
                                                    result_dict, epoch, verbose)
            elif learn_algo == 'ppo':
                result_dict = self._train_ppo(episodes, problem_size, batch_size, n_rollouts, max_clip_norm, start_time,
                                              result_dict, epoch, verbose)
            else:
                raise NotImplementedError

            # Evaluate the model
            if (eval_freq>0) and (epoch + 1) % eval_freq == 0:
                elapsed_time = time.time() - start_time
                obj_values = self._fast_eval(eval_state, eval_obj_value)
                result_dict['eval_obj_values'].append(obj_values.mean().item())
                result_dict['eval_elapsed_times'].append(elapsed_time)
                if verbose:
                    print(f"Epoch {epoch + 1}, Eval Obj value: {obj_values.mean().item():.3f}, Elapsed Time: {elapsed_time:.2f} sec")

            # Save the model every save_freq epochs
            if (save_freq>0) and ((epoch + 1) % save_freq == 0 or (epoch + 1 == epochs)):
                saved_path = self.save_checkpoint(cur_run_name, epoch+1)
                if verbose:
                    print(f"Saving checkpoint in {saved_path}.")

        # End of training loop
        if verbose:
            print(f"End of model training. Total Elapsed Time: {time.time() - start_time:.2f} sec")

        return result_dict

    def _train_reinforce(self, episodes, problem_size, batch_size, n_rollouts, max_clip_norm, start_time,
                         result_dict, epoch, verbose):

        # train loop
        running_obj_avg = 0
        pomo_size = 1
        for episode in range(episodes):
            # Reset the environment
            state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size, pomo_size=pomo_size)

            # Get logits (shape: batch_size x (num_nodes or num_edges) x output_dim)
            logits = self.model(state)

            # Mask invalid actions
            if state.mask is not None:
                logits = logits + state.mask.reshape(batch_size * pomo_size, logits.size(1), -1)

            # Decode solutions in the environment
            reward, log_prob, obj_value = self.env.heatmap_decode(logits, n_rollouts=n_rollouts, deterministic=False)

            # Once an episode is done, calculate the loss
            # Leave one out baseline
            sum_r = reward.sum(dim=0, keepdim=True)
            baselines_loo = (sum_r - reward) / (reward.shape[0] - 1)
            advantage = reward - baselines_loo

            loss = -advantage * log_prob  # Minus Sign: To Increase REWARD

            # Calculate the mean loss and back-propagate
            loss_mean = loss.mean()
            self.optimizer.zero_grad()
            loss_mean.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm) # Clip gradients
            self.optimizer.step() # Update the model

            # Append results
            elapsed_time = time.time() - start_time
            obj_mean = obj_value.mean().item()
            result_dict['loss_values'].append(loss_mean.item())
            result_dict['obj_values'].append(obj_mean)
            result_dict['elapsed_times'].append(elapsed_time)

            if verbose:
                running_obj_avg += obj_mean
                print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss_mean.item():.4f}, Obj value: {obj_mean:.3f} (running avg {running_obj_avg/(episode+1):.3f}), Elapsed Time: {elapsed_time:.2f} sec")
        return result_dict

    def _train_ppo(self, episodes, problem_size, batch_size, n_rollouts, max_clip_norm, start_time,
                   result_dict, epoch, verbose):
        # TODO
        raise NotImplementedError


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

        # Get logits
        logits = self.model(state)
        if state.mask is not None:
            logits = logits + state.mask.reshape(state.batch_size*state.pomo_size, logits.size(1), -1)  # mask invalid actions

        # Take a step in the environment
        reward, log_prob, obj_value = self.env.heatmap_decode(logits, n_rollouts=state.pomo_size, deterministic=True)

        return obj_value


class ImprovementTrainer(Trainer):
    def __init__(self, model: nn.Module, env: Env, optimizer: torch.optim.Optimizer, device: torch.device):
        """
        Trainer class for improvement problems.
        """
        super().__init__(model, env, optimizer, device)

    @torch.no_grad()
    def inference(self, problem_size: int or None = None, batch_size: int or None = None, pomo_size: int = 1, deterministic: bool = True,
                  use_custom_instances: bool = False, seed: int or None = None, verbose: bool = False) -> (torch.Tensor, dict):
        """
        Run the improvement model in inference mode.
        :param problem_size: The size of the problem. Type: int or None.
        :param batch_size: The size of the batch. Type: int or None.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param deterministic: If True, choose the action with the highest probability. Type: bool.
        :param use_custom_instances: If True, custom instances will be used. Type: bool.
        :param seed: The seed for reproducibility. Type: int or None.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.

        :return: The objective value of the proposed solutions. Type: torch.Tensor.
        :return: The result dictionary. Type: dict.
        """

        # Improvement trainer, therefore, the reward and stopping criteria should be improvement also
        assert self.env.reward.__class__.__name__ == 'ImprovementReward', "The reward class should be ImprovementReward"
        assert self.env.stopping_criteria.__class__.__name__ == 'ImprovementStoppingCriteria', "The stopping criteria should be ImprovementStoppingCriteria"

        if not use_custom_instances:
            assert problem_size is not None, "The problem size should be provided"
            assert batch_size is not None, "The batch size should be provided"

        # Start timer and result dictionary
        start_time = time.time()
        result_dict = {
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
                                                        pomo_size=pomo_size, use_custom_instances=use_custom_instances,
                                                        seed=seed)

        # Check if batch_size, pomo_size or problem_size have been modified while using custom instances
        batch_size = state.batch_size
        pomo_size = state.pomo_size
        problem_size = state.problem_size

        # Main inference loop
        step = 0
        best_obj_values = obj_value.max(dim=1).values
        while not done:
            step += 1

            # Get logits
            logits = self.model(state)

            # Mask invalid actions
            if state.mask is not None:
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
              eval_problem_size: int, eval_batch_size: int, learn_algo: str = 'reinforce', baseline_type: str = 'mean',
              update_freq: int = 10, gamma: float = 0.99, max_clip_norm: float = 1.0, ppo_args: dict = None, eval_freq: int = 1,
              save_freq: int = -1, save_path_name: str = '', seed: int = 42, verbose: bool = False) -> dict:
        """
        Train the improvement model.
        :param epochs: The number of epochs to train. Type: int.
        :param episodes: The number of episodes to train in each epoch. Type: int.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param pomo_size: Number of parallel initializations (Policy Optimization with Multiple Optima). Type: int.
        :param eval_problem_size: The size of the evaluation problem. Type: int.
        :param eval_batch_size: The size of the evaluation batch. Type: int.
        :param learn_algo: Learning Algorithm to use. Options: 'reinforce', 'ppo'. Type: str.
        :param baseline_type: The baseline type to use. Options: 'mean', 'scst', 'pomo' Type: str.
        :param update_freq: The frequency of updating the model. Type: int.
        :param gamma: The discount factor to determine the importance of future rewards. Type: float.
        :param max_clip_norm: The maximum norm for clipping gradients. Type: float.
        :param ppo_args: Additional arguments to pass to the PPO algorithm. Type: dict or None.
        :param eval_freq: The frequency (in epochs) of evaluating the model. Type: int.
        :param save_freq: The frequency (in epochs) of saving the model. Type: int.
        :param save_path_name: The path to save the model. Type: str.
        :param seed: The seed for reproducibility. Type: int.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.
        """
        # Test the problem definition
        test_problem_def(self.model, self.env, problem_size, batch_size, pomo_size)

        # Initialize the run name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cur_run_name = f"NC_train_{save_path_name}_{timestamp}"

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

        # Training loop
        for epoch in range(epochs):
            # Set the model to training mode
            self.model.train()
            if learn_algo == 'reinforce':
                result_dict = self._train_reinforce(episodes, problem_size, batch_size, pomo_size, gamma, update_freq,
                                                    max_clip_norm, start_time, result_dict, epoch, verbose)
            elif learn_algo == 'ppo':
                result_dict = self._train_ppo(episodes, problem_size, batch_size, pomo_size, gamma, ppo_args, max_clip_norm,
                                              start_time, result_dict, epoch, verbose)
            else:
                raise NotImplementedError

            # Evaluate the model
            if (eval_freq>0) and ((epoch + 1) % eval_freq == 0):
                elapsed_time = time.time() - start_time
                obj_values = self._fast_eval(eval_state, eval_obj_value)
                result_dict['eval_obj_values'].append(obj_values.mean().item())
                result_dict['eval_elapsed_times'].append(elapsed_time)
                if verbose:
                    print(f"Epoch {epoch + 1}, Eval Obj value: {obj_values.mean().item():.3f}, Elapsed Time: {elapsed_time:.2f} sec")

            # Save the model every save_freq epochs
            if (save_freq>0) and ((epoch + 1) % save_freq == 0 or (epoch + 1 == epochs)):
                saved_path = self.save_checkpoint(cur_run_name, epoch+1)
                if verbose:
                    print(f"Saving checkpoint in {saved_path}.")

        # End of training loop
        if verbose:
            print(f"End of model training. Total Elapsed Time: {time.time() - start_time:.2f} sec")
        return result_dict

    def _train_reinforce(self, episodes, problem_size, batch_size, pomo_size, gamma, update_freq, max_clip_norm, start_time,
                         result_dict, epoch, verbose):
        # Initialize the batch range
        batch_pomo_range = torch.arange(batch_size*pomo_size)

        avg_loss = 0
        running_obj_avg = 0
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
                    logits = self.model(state)

                    # Mask invalid actions
                    if state.mask is not None:
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
                    rewards.append(reward.reshape(batch_size * pomo_size))

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
                running_obj_avg += best_obj_value.mean().item()
                update_count += 1

                # Clip grad norms
                nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                # Update the model
                self.optimizer.step()

            if verbose:
                print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {avg_loss / update_count:.4f}, "
                      f"Reward: {reward.mean().item():.3f}, Obj value: {running_obj_avg / update_count:.3f}, "
                      f"Best Obj value: {best_obj_value.mean().item():.3f}, "
                      f"Steps: {episode_steps}, Elapsed Time: {time.time() - start_time:.2f} sec")

            # Append results
            result_dict['loss_values'].append(avg_loss / update_count)
            result_dict['obj_values'].append(running_obj_avg / update_count)
            result_dict['elapsed_times'].append(time.time() - start_time)

        return result_dict

    def _train_ppo(self, episodes, problem_size, batch_size, pomo_size, gamma, ppo_args, max_clip_norm, start_time, result_dict, epoch, verbose):

        #ppo_epochs, ppo_clip, entropy_coef, ppo_update_batch_count, n_stored_states = ppo_args
        # TODO use as a dict

        # Initialize the batch range
        batch_pomo_range = torch.arange(batch_size*pomo_size)

        running_obj_avg = 0
        obj_count = 0
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        # Run episodes
        for episode in range(episodes):
            # Reset the environment
            state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size,
                                                            pomo_size=pomo_size)

            # Initialize the best objective value found so far
            best_obj_value = obj_value.clone()

            # Episode loop
            states_list = []
            actions_list = []
            old_log_probs_list = []
            rewards_list = []
            episode_steps = 0
            while not done:
                states_list.append(state)
                episode_steps += 1

                # Get logits
                with torch.no_grad():
                    logits = self.model(state)

                    # Mask invalid actions
                    if state.mask is not None:
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

                # Store transitions
                actions_list.append(action)
                old_log_probs_list.append(prob.log().detach())
                rewards_list.append(reward.detach())

            running_obj_avg += best_obj_value.mean().item()
            obj_count += 1

            discounted_sum = torch.zeros(batch_size, pomo_size, device=self.device)
            ep_returns = []  # will hold R_t at index t for this episode

            for r in reversed(rewards_list):
                discounted_sum = r + gamma * discounted_sum
                ep_returns.insert(0, discounted_sum.clone())
            # Now ep_returns[0] is return-at-time-0, ep_returns[1] is time-1, …, in episode-order

            # Compute advantages: advantage[t] = returns[t] − baseline[t],
            #    where baseline[t] = average over pop dimension
            returns_stack = torch.stack(ep_returns, dim=0)  # [steps, batch, pop]
            advantages_list = []
            for t in range(episode_steps):
                returns_t = returns_stack[t]  # [batch, pop]
                # Leave one out baseline
                if pomo_size > 1:
                    sum_r = returns_t.sum(dim=1, keepdim=True)
                    baseline_t = (sum_r - returns_t) / (pomo_size - 1)
                    adv_t = returns_t - baseline_t  # [batch, pop]
                else:
                    adv_t = returns_t
                advantages_list.append(adv_t)

            # Package episode data
            n_to_store = min(ppo_args['n_stored_states'], episode_steps)

            if n_to_store >= 2:
                # we want to include first and last, and evenly distribute the rest
                step = (episode_steps - 1) / (n_to_store - 1)
                indices = [int(round(i * step)) for i in range(n_to_store)]
            elif n_to_store == 1:
                # only one state to pick
                indices = [0]
            else:
                # no steps at all
                indices = []

            # Subsample the states
            all_states.extend([states_list[i] for i in indices])  # [A₀, A₁, …, B₀, B₁, …]
            all_actions.extend([actions_list[i] for i in indices])
            all_old_log_probs.extend([old_log_probs_list[i] for i in indices])
            all_advantages.extend([advantages_list[i] for i in indices])

            if verbose:
                print(f"Epoch {epoch + 1}, Episode {episode + 1}, "
                      f"Reward: {reward.mean().item():.3f}, Obj value: {running_obj_avg / obj_count:.3f}, "
                      f"Best Obj value: {best_obj_value.mean().item():.3f}, "
                      f"Steps: {episode_steps}, Elapsed Time: {time.time() - start_time:.2f} sec")

            # Append results
            result_dict['obj_values'].append(running_obj_avg / obj_count)
            result_dict['elapsed_times'].append(time.time() - start_time)

        # Update weights with PPO
        T_total = len(all_states)  # total number of time steps across buffer: n_episodes * n_steps (each episode)

        # Perform PPO epochs
        total_loss_accum = 0.0
        total_entropy_accum = 0.0
        n_updates = 0
        idxs = list(range(T_total))
        for _ in range(ppo_args['ppo_epochs']):
            random.shuffle(idxs)
            epoch_loss = 0.0
            epoch_entropy_sum = 0.0
            batch_count = 0
            for t in range(T_total):
                cur_idx = idxs[t]
                state_t = all_states[cur_idx]  # a single “state” object
                actions_t = all_actions[cur_idx][0].view(batch_size*pomo_size, -1)  # [batch*pop]
                old_logp_t = all_old_log_probs[cur_idx]  # [batch*pop]
                adv_t = all_advantages[cur_idx].reshape(-1)  # [batch*pop]

                # forward
                new_logits = self.model(state_t)  # [batch*pop, action_dim]
                new_logits = new_logits.reshape(batch_size * pomo_size, -1)  # reshape logits
                new_logp = F.log_softmax(new_logits, dim=-1)  # [batch*pop]
                new_logp_actions = new_logp.gather(1, actions_t).squeeze(-1)

                if ppo_args['entropy_coef'] == 0:
                    with torch.no_grad():
                        probs = new_logp.exp()
                else:
                    probs = new_logp.exp()

                safe_log_p = torch.nan_to_num(new_logp, nan=0.0, neginf=0.0, posinf=0.0)
                entropy = -(safe_log_p * probs).sum(dim=-1).mean()  # [1]

                ratio = torch.exp(new_logp_actions - old_logp_t)
                surrogate = torch.min(
                    ratio * adv_t, torch.clamp(ratio, 1-ppo_args['ppo_clip'], 1+ppo_args['ppo_clip']) * adv_t
                ).mean()

                loss = -surrogate - ppo_args['entropy_coef'] * entropy

                epoch_loss += loss
                epoch_entropy_sum += entropy.item()
                batch_count += 1
                if (batch_count >= ppo_args['ppo_update_batch_count']) or (t == T_total - 1):
                    # backward & step
                    self.optimizer.zero_grad()
                    epoch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)
                    self.optimizer.step()

                    total_loss_accum += epoch_loss.item()
                    total_entropy_accum += epoch_entropy_sum

                    # free graph
                    torch.cuda.empty_cache()

                    epoch_loss = 0.0
                    epoch_entropy_sum = 0.0
                    batch_count = 0
                    n_updates += 1

        result_dict['loss_values'].append(total_loss_accum / n_updates)

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
            logits = self.model(state)

            # Mask invalid actions
            if state.mask is not None:
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
    logits = model(cur_state)
    if cur_state.mask is not None:
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
