"""
Author:      Yi Zhang, Master Student @ idrugLab, School of Biology and Biological Engineering, South China Universty of Technology
Created on:  2022/11/17
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal


class GAIL:
    def __init__(self, models, memory, cfg):
        self.gamma = cfg.gamma
        self.continuous = cfg.continuous
        if hasattr(cfg, 'action_bound'):
            self.action_bound = cfg.action_bound
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.batch_size = cfg.batch_size
        self.gae_lambda = cfg.gae_lambda
        self.device = torch.device(cfg.device)
        self.actor = models['Actor'].to(self.device)
        self.critic = models['Critic'].to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = memory
        self.loss = 0

    def sample_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        if self.continuous:
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.view(1, ), sigma.view(1, ))
            action = dist.sample()
            value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            value = value.detach().cpu().numpy().squeeze(0).item()  # detach() to avoid gradient
            log_probs = dist.log_prob(action).item()  # Tensor([0.])
            # self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.cpu().detach().numpy(), log_probs, value
        else:
            probs = self.actor(state)
            dist = Categorical(probs)
            value = self.critic(state)
            action = dist.sample()
            probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            value = torch.squeeze(value).item()
            return action, probs, value

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        if self.continuous:
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.view(1, ), sigma.view(1, ))
            action = dist.sample()
            value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            # value = value.detach().cpu().numpy().squeeze(0)[0] # detach() to avoid gradient
            log_probs = dist.log_prob(action).item()  # Tensor([0.])
            # self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.cpu().numpy(), log_probs, value.cpu()
        else:
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            value = torch.squeeze(value).item()
            return action, probs, value

    def update(self):
        # states = policy_trajectory_replays['states']
        # actions = policy_trajectory_replays['actions']
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            ### compute advantage ###
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                probs = self.actor(states)
                dist = Categorical(probs)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        actor_checkpoint = os.path.join(fpath, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(fpath, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)

    def load_model(self, fpath):
        actor_checkpoint = torch.load(os.path.join(fpath, 'ppo_actor.pt'), map_location=self.device)
        critic_checkpoint = torch.load(os.path.join(fpath, 'ppo_critic.pt'), map_location=self.device)
        self.actor.load_state_dict(actor_checkpoint)
        self.critic.load_state_dict(critic_checkpoint)
