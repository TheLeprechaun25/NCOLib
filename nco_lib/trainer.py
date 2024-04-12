from abc import ABC, abstractmethod
import time
import torch
import torch.nn as nn
import torch.nn.functional as F



class Trainer(ABC):
    def __init__(self, model, env, optimizer, device):
        self.model = model
        self.env = env
        self.optimizer = optimizer
        self.device = device

    @abstractmethod
    def inference(self, problem_size, batch_size):
        pass

    @abstractmethod
    def train(self, epochs, episodes, problem_size, batch_size, max_clip_norm=1.0, save_freq=10, save_path=''):
        pass

    def save_checkpoint(self, path, epoch):
        model_path = path + '_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']


class ConstructiveTrainer(Trainer):
    @torch.no_grad()
    def inference(self, problem_size=20, batch_size=1, deterministic=True, seed=None, verbose=False):
        self.model.eval()
        if verbose:
            print("\nStart of model inference")
        state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size, seed=seed)
        while not done:
            logits = self.model(state)  # get logits
            logits = logits + state.mask  # mask invalid actions
            logits = logits.reshape(batch_size, -1)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1) # softmax to get probabilities
                action = probs.multinomial(1).squeeze(dim=1)  # sample action from the distribution

            state, reward, obj_value, done = self.env.step(action)
        if verbose:
            print(f"Objective Value: {obj_value.mean().item()}")

        return obj_value

    def train(self, epochs, episodes, problem_size, batch_size, max_clip_norm=1.0, save_freq=10, save_path='', verbose=False):
        self.model.train()
        if verbose:
            print("\nStart of model training")
        batch_range = torch.arange(batch_size)
        start_time = time.time()
        for epoch in range(epochs):
            for episode in range(episodes):
                state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size)
                prob_list = torch.zeros(size=(batch_size, 0), device=self.device)
                self.optimizer.zero_grad()
                while not done:
                    logits = self.model(state)
                    # mask invalid actions TODO: check shape of mask
                    logits = logits + state.mask

                    logits = logits.reshape(batch_size, -1)

                    # softmax to get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    action = probs.multinomial(1).squeeze(dim=1)
                    prob = probs[batch_range, action]
                    prob_list = torch.cat((prob_list, prob[:, None]), dim=1)

                    state, reward, obj_value, done = self.env.step(action)

                log_prob = prob_list.log().sum(dim=1)
                # TODO change RL algorithm and baseline
                advantage = reward - reward.mean()
                loss = -advantage * log_prob  # Minus Sign: To Increase REWARD

                loss_mean = loss.mean()
                loss_mean.backward()
                # clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                self.optimizer.step()
                if verbose:
                    print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss_mean.item():.4f}, Obj value: {obj_value.mean().item():.3f}, Elapsed Time: {time.time() - start_time:.2f} sec")

            if save_path and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_path, epoch)


class ImprovementTrainer(Trainer):
    @torch.no_grad()
    def inference(self, problem_size=20, batch_size=1, deterministic=True, seed=None, verbose=False):
        self.model.eval()
        if verbose:
            print("\nStart of model inference")
        state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size, seed=seed)
        step = 0
        while not done:
            step += 1
            logits = self.model(state)  # get logits
            logits = logits + state.mask  # mask invalid actions
            logits = logits.reshape(batch_size, -1)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)  # softmax to get probabilities
                action = probs.multinomial(1).squeeze(dim=1)

            state, reward, obj_value, done = self.env.step(action)
            if verbose:
                print(f"Step {step}. Obj value: {obj_value.mean().item()}")

    def train(self, epochs, episodes, problem_size, batch_size, update_freq=10, gamma=0.99, max_clip_norm=1.0, save_freq=10, save_path='', verbose=False):
        self.model.train()
        if verbose:
            print("\nStart of model training")
        batch_range = torch.arange(batch_size)
        start_time = time.time()
        for epoch in range(epochs):
            for episode in range(episodes):
                state, reward, obj_value, done = self.env.reset(problem_size=problem_size, batch_size=batch_size)
                best_obj_value = obj_value.clone()
                while not done:
                    log_probs = []
                    rewards = []
                    for step in range(update_freq):
                        logits = self.model(state)
                        # mask invalid actions
                        logits = logits + state.mask

                        logits = logits.reshape(batch_size, -1)

                        # softmax to get probabilities
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        action = probs.multinomial(1).squeeze(dim=1)
                        prob = probs[batch_range, action]

                        state, reward, obj_value, done = self.env.step(action)

                        best_obj_value = torch.max(best_obj_value, obj_value)

                        log_probs.append(prob.log())
                        rewards.append(reward)

                        if done:
                            break

                    # Calculate discounting rewards
                    t_steps = torch.arange(len(rewards))
                    discounts = gamma ** t_steps
                    r = [r_i * d_i for r_i, d_i in zip(rewards, discounts)]
                    r = r[::-1]
                    b = torch.cumsum(torch.stack(r), dim=0)
                    c = [b[k, :] for k in reversed(range(b.shape[0]))]
                    R = [c_i / d_i for c_i, d_i in zip(c, discounts)]

                    self.optimizer.zero_grad()
                    policy_loss = []
                    for log_prob_i, r_i in zip(log_probs, R):
                        policy_loss.append(-log_prob_i * r_i)
                    loss = torch.cat(policy_loss).mean()
                    loss.backward()

                    # clip grad norms
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                    self.optimizer.step()


                    if verbose:
                        print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss.item():.4f}, "
                              f"Reward: {reward.mean().item():.3f}, Obj value: {best_obj_value.mean().item():.3f}, "
                              f"Elapsed Time: {time.time() - start_time:.2f} sec")

            if save_path and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_path, epoch)


class NARHeatmapTrainer(Trainer):
    @torch.no_grad()
    def inference(self, problem_size=20, batch_size=1, deterministic=True, seed=None, verbose=False):
        self.model.eval()
        state, obj_value, _ = self.env.reset(problem_size, batch_size, seed=seed)

        logits = self.model(state)  # get logits
        logits = logits + state.mask  # mask invalid actions
        logits = logits.reshape(batch_size, -1)

        state, obj_value, _ = self.env.step(logits, deterministic=deterministic)

        if verbose:
            print(f"Objective Value: {obj_value.mean().item()}")

        return obj_value

    def train(self, epochs, episodes, problem_size, batch_size, max_clip_norm=1.0, save_freq=10, save_path='', verbose=False):
        self.model.train()
        batch_range = torch.arange(batch_size)
        start_time = time.time()
        for epoch in range(epochs):
            for episode in range(episodes):
                state, obj_value, done = self.env.reset(problem_size, batch_size)
                prob_list = torch.zeros(size=(batch_size, 0), device=self.device)
                self.optimizer.zero_grad()
                while not done:
                    logits = self.model(state)
                    # mask invalid actions
                    logits = logits + state.mask

                    logits = logits.reshape(batch_size, -1)

                    # softmax to get probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    action = probs.multinomial(1).squeeze(dim=1)
                    prob = probs[batch_range, action]
                    prob_list = torch.cat((prob_list, prob[:, None]), dim=1)

                    state, obj_value, done = self.env.step(action)

                log_prob = prob_list.log().sum(dim=1)
                # TODO change RL algorithm and baseline
                advantage = obj_value - obj_value.mean()
                loss = -advantage * log_prob  # Minus Sign: To Increase REWARD

                loss_mean = loss.mean()
                loss_mean.backward()
                # clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                self.optimizer.step()
                if verbose:
                    print(f"Epoch {epoch + 1}, Episode {episode + 1}, Loss: {loss_mean.item():.4f}, Obj value: {obj_value.mean().item():.3f}, Elapsed Time: {time.time() - start_time:.2f} sec")

            if save_path and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_path, epoch)

