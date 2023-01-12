#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-08-16 23:05:25
LastEditor: JiangJi
LastEditTime: 2022-12-04 14:44:03
Discription: 
'''
import torch
import numpy as np
from torch.distributions import Categorical,Normal
from common.models import ActorSoftmax, Critic
from common.memories import PGReplay
        
class Agent:
    def __init__(self,cfg):
        self.n_actions = cfg.n_actions
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef
        self.device = torch.device(cfg.device) 
        self.continuous = cfg.continuous
        if hasattr(cfg,'action_bound'):
            self.action_bound = cfg.action_bound
        self.actor = ActorSoftmax(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states,1,hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.sample_count = 0
        self.update_freq = cfg.update_freq
    def sample_action(self,state):
        # state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        # dist = self.actor(state)
        # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
        # value = self.critic(state) # note that 'dist' need require_grad=True
        # self.value = value.detach().cpu().numpy().squeeze(0)[0]
        # action = np.random.choice(self.n_actions, p=dist.detach().cpu().numpy().squeeze(0)) # shape(p=(n_actions,1)
        # self.log_prob = torch.log(dist.squeeze(0)[action])
        self.sample_count += 1
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.view(1,), sigma.view(1,))
            action = dist.sample()
            value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            self.value = value.detach().cpu().numpy().squeeze(0)[0] # detach() to avoid gradient
            self.log_prob = dist.log_prob(action).squeeze(dim=0) # Tensor([0.])
            self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.cpu().detach().numpy()
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            probs = self.actor(state)
            dist = Categorical(probs)
            action = dist.sample() # Tensor([0])
            value = self.critic(state)
            return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self,state):
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.view(1,), sigma.view(1,))
            action = dist.sample()
            return action.cpu().detach().numpy()
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            probs = self.actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            return action.detach().cpu().numpy().item()
    def update(self,next_state):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        states, actions, rewards, dones = self.memory.sample()
        # convert to tensor
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32)
        # compute returns
        if next_state is not None:
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            next_value = self.critic(next_state).detach().cpu().numpy().squeeze(0)[0]
        else:
            next_value = 0 # terminal state
        returns = self.compute_returns(next_value,rewards,dones)
        values = self.critic(states)
        advantages = returns - values.detach()
        probs = self.actor(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions).unsqueeze(dim=1) # log_probs.shape = (batch_size,1), which is the same as advantages.shape
        actor_loss = (-log_probs*advantages).mean()+ self.entropy_coef * dist.entropy().mean()
        # critic_loss = (0.5 * advantages).pow(2).mean()
        # tot_loss = actor_loss + critic_loss
        # self.actor_optimizer.zero_grad()
        # self.critic_optimizer.zero_grad()
        # tot_loss.backward()
        # self.actor_optimizer.step()
        # self.critic_optimizer.step()
        critic_loss = (returns-values).pow(2).mean()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        # clear memory
        self.memory.clear()
    def compute_returns(self, next_value, rewards, dones):
        '''monte carlo estimate of state rewards'''
        returns = torch.zeros_like(rewards)
        R = next_value
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32).unsqueeze(1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by 
        return returns
        
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{fpath}/actor_checkpoint.pt")
        torch.save(self.critic.state_dict(), f"{fpath}/critic_checkpoint.pt")

    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/actor_checkpoint.pt")
        critic_ckpt = torch.load(f"{fpath}/critic_checkpoint.pt")
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)