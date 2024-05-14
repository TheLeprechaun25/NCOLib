from abc import ABC, abstractmethod
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment.env import Env


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
        self.env = env
        self.optimizer = optimizer
        self.device = device

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

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """
        Save the model and optimizer state.
        :param path: The path to save the model. Type: str.
        :param epoch: The epoch number. Type: int.
        """
        model_path = path + '_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

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
    def inference(self, problem_size: int = 20, batch_size: int = 1, deterministic: bool = True,
                  seed: int or None = None, verbose: bool = False) -> torch.Tensor:
        """
        Run the constructive model in inference mode.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param deterministic: If True, choose the action with the highest probability. Type: bool.
        :param seed: The seed for reproducibility. Type: int or None.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.
        :return: The objective value of the proposed solutions. Type: torch.Tensor.
        """
        # Set the model to evaluation mode
        self.model.eval()

        if verbose:
            print("\nStart of model inference")

        # Reset the environment
        state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size, seed=seed)

        # Main loop
        start_time = time.time()
        while not done:
            # Get logits
            logits = self.model(state)
            logits = logits + state.mask  # mask invalid actions
            logits = logits.reshape(batch_size, -1)  # reshape logits

            # Get actions from the model logits
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1) # softmax to get probabilities
                action = probs.multinomial(1).squeeze(dim=1)  # sample action from the distribution

            # Take a step in the environment
            state, reward, obj_value, done = self.env.step(action)

        # End of inference loop
        if verbose:
            print(f"Objective Value: {obj_value.mean().item()}. Elapsed Time: {time.time() - start_time:.2f} sec")

        return obj_value

    def train(self, epochs: int, episodes: int, problem_size: int, batch_size: int, max_clip_norm: float = 1.0,
              save_freq: int = 10, save_path: str = '', verbose: bool = False) -> None:
        """
        Train the constructive model.
        :param epochs: The number of epochs to train. Type: int.
        :param episodes: The number of episodes to train in each epoch. Type: int.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param max_clip_norm: The maximum norm for clipping gradients. Type: float.
        :param save_freq: The frequency of saving the model. Type: int.
        :param save_path: The path to save the model. Type: str.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.
        """
        # Set the model to training mode
        self.model.train()

        if verbose:
            print("\nStart of model training")

        # Initialize the batch range
        batch_range = torch.arange(batch_size)

        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            for episode in range(episodes):
                # Reset the environment
                state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size)

                # Initialize the probability list
                prob_list = torch.zeros(size=(batch_size, 0), device=self.device)

                # Main inference loop
                self.optimizer.zero_grad()
                while not done:
                    # Get logits
                    logits = self.model(state)

                    # Mask invalid actions
                    logits = logits + state.mask
                    logits = logits.reshape(batch_size, -1)  # reshape logits

                    # Get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # Sample action from the distribution
                    action = probs.multinomial(1).squeeze(dim=1)

                    # Get the probability of the selected action and append it to the list
                    prob = probs[batch_range, action]
                    prob_list = torch.cat((prob_list, prob[:, None]), dim=1)

                    # Take a step in the environment
                    state, reward, obj_value, done = self.env.step(action)

                # Once episode is done, calculate the loss
                log_prob = prob_list.log().sum(dim=1)
                advantage = reward - reward.mean()
                loss = -advantage * log_prob  # Minus Sign: To Increase REWARD

                # Calculate the mean loss and backpropagate
                loss_mean = loss.mean()
                loss_mean.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                # Update the model
                self.optimizer.step()

                if verbose:
                    print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss_mean.item():.4f}, Obj value: {obj_value.mean().item():.3f}, Elapsed Time: {time.time() - start_time:.2f} sec")

            # Save the model every save_freq epochs
            if save_path and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_path, epoch)

        # End of training loop - Save the final model
        self.save_checkpoint(save_path, epochs)

        if verbose:
            print(f"End of model training. Model saved. Total Elapsed Time: {time.time() - start_time:.2f} sec")


class ImprovementTrainer(Trainer):
    def __init__(self, model: nn.Module, env: Env, optimizer: torch.optim.Optimizer, device: torch.device):
        """
        Trainer class for improvement problems.
        """
        super().__init__(model, env, optimizer, device)

    @torch.no_grad()
    def inference(self, problem_size: int = 20, batch_size: int = 1, deterministic: bool = True,
                  seed: int or None = None, verbose: bool = False):
        """
        Run the improvement model in inference mode.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param deterministic: If True, choose the action with the highest probability. Type: bool.
        :param seed: The seed for reproducibility. Type: int or None.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.
        """
        # Set the model to evaluation mode
        self.model.eval()

        if verbose:
            print("\nStart of model inference")

        # Reset the environment
        state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size, seed=seed)

        # Main inference loop
        step = 0
        while not done:
            step += 1

            # Get logits
            logits = self.model(state)

            # Mask invalid actions
            logits = logits + state.mask
            logits = logits.reshape(batch_size, -1)  # reshape logits

            # Get actions from the model logits
            if deterministic:
                action = logits.argmax(dim=-1)  # choose the action with the highest probability
            else:
                probs = F.softmax(logits, dim=-1)  # softmax to get probabilities
                action = probs.multinomial(1).squeeze(dim=1)  # sample action from the distribution

            # Take a step in the environment
            state, reward, obj_value, done = self.env.step(action)

            if verbose:
                print(f"Step {step}. Obj value: {obj_value.mean().item()}")

        # End of inference loop
        if verbose:
            print(f"Objective Value: {obj_value.mean().item()}")

        return obj_value

    def train(self, epochs: int, episodes: int, problem_size: int, batch_size: int, update_freq: int = 10,
              gamma: float = 0.99, max_clip_norm: float = 1.0, save_freq: int = 10, save_path: str = '',
              verbose: bool = False) -> None:
        """
        Train the improvement model.
        :param epochs: The number of epochs to train. Type: int.
        :param episodes: The number of episodes to train in each epoch. Type: int.
        :param problem_size: The size of the problem. Type: int.
        :param batch_size: The size of the batch. Type: int.
        :param update_freq: The frequency of updating the model. Type: int.
        :param gamma: The discount factor to determine the importance of future rewards. Type: float.
        :param max_clip_norm: The maximum norm for clipping gradients. Type: float.
        :param save_freq: The frequency of saving the model. Type: int.
        :param save_path: The path to save the model. Type: str.
        :param verbose: If True, print the loss, objective value, and elapsed time. Type: bool.
        """
        # Set the model to training mode
        self.model.train()

        if verbose:
            print("\nStart of model training")

        # Initialize the batch range
        batch_range = torch.arange(batch_size)

        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            for episode in range(episodes):
                # Reset the environment
                state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size)

                # Initialize the best objective value found so far
                best_obj_value = obj_value.clone()

                # Episode loop
                while not done:
                    # Update loop
                    log_probs = []
                    rewards = []
                    for step in range(update_freq):
                        # Get logits
                        logits = self.model(state)

                        # Mask invalid actions
                        logits = logits + state.mask
                        logits = logits.reshape(batch_size, -1)

                        # Softmax to get probabilities
                        probs = torch.nn.functional.softmax(logits, dim=-1)

                        # Sample action from the distribution
                        action = probs.multinomial(1).squeeze(dim=1)
                        prob = probs[batch_range, action]

                        # Take a step in the environment
                        state, reward, obj_value, done = self.env.step(action)

                        # Update the best objective value found so far
                        best_obj_value = torch.max(best_obj_value, obj_value)

                        # Append log probabilities and rewards
                        log_probs.append(prob.log())
                        rewards.append(reward)

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

                    # Clip grad norms
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                    # Update the model
                    self.optimizer.step()

                    if verbose:
                        print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss.item():.4f}, "
                              f"Reward: {reward.mean().item():.3f}, Obj value: {best_obj_value.mean().item():.3f}, "
                              f"Elapsed Time: {time.time() - start_time:.2f} sec")

            if save_path and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_path, epoch)

        # End of training loop - Save the final model
        self.save_checkpoint(save_path, epochs)
