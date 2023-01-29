#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-09-26 16:11:36
LastEditor: JiangJi
LastEditTime: 2022-12-03 16:47:12
Discription:  #TODO,保存留作GAE计算
'''

import os
import numpy as np
import torch 
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from functools import partial
from collections import deque


def mini_batch(batch, mini_batch_size):
    mini_batch_size += 1
    states, actions, old_log_probs, adv, td_target = zip(*batch)
    return torch.stack(states[:mini_batch_size]), torch.stack(actions[:mini_batch_size]), \
        torch.stack(old_log_probs[:mini_batch_size]), torch.stack(adv[:mini_batch_size]), torch.stack(td_target[:mini_batch_size])


class memDataset(Dataset):
    def __init__(self, states: tensor, actions: tensor, old_log_probs: tensor, 
                 advantage: tensor, td_target: tensor):
        super(memDataset, self).__init__()
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantage = advantage
        self.td_target = td_target
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        states = self.states[index]
        actions = self.actions[index]
        old_log_probs = self.old_log_probs[index]
        adv = self.advantage[index]
        td_target = self.td_target[index]
        return states, actions, old_log_probs, adv, td_target



class PPO:
    def __init__(self, models,memory,cfg):
        self.gamma = cfg.gamma
        self.continuous = cfg.continuous
        self.action_bound = 1.0
        if hasattr(cfg,'action_bound'):
            self.action_bound = cfg.action_bound
        self.policy_clip = cfg.policy_clip
        self.k_epochs = cfg.k_epochs
        # self.batch_size = cfg.batch_size
        self.gae_lambda = cfg.gae_lambda
        self.device = torch.device(cfg.device) 
        self.actor = models['Actor'].to(self.device)
        self.critic = models['Critic'].to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = memory
        self.loss = 0
        
        self.sgd_batch_size = cfg.sgd_batch_size
        self.minibatch_size = cfg.minibatch_size
        self.min_batch_collate_func = partial(mini_batch, mini_batch_size=self.minibatch_size)
    
    def sample_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        if self.continuous:
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.detach(), sigma.detach())
            action = dist.sample()
            value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            log_probs = dist.log_prob(action) # Tensor([0.])
            # self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            self.value = value.cpu().detach().numpy()[0]
            self.log_probs = log_probs.detach().numpy()[0]
            return action.detach().numpy()[0], self.log_probs, self.value
        else:
            probs = self.actor(state)
            dist = Categorical(probs)
            value = self.critic(state)
            action = dist.sample()
            probs = dist.log_prob(action).detach().numpy()
            action = action.detach().numpy()
            value = value.detach().numpy()
            self.value = value
            self.log_probs = probs
            return action, probs, value

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        if self.continuous:
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.detach(), sigma.detach())
            action = dist.sample()
            value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            # value = value.detach().cpu().numpy().squeeze(0)[0] # detach() to avoid gradient
            log_probs = dist.log_prob(action) # Tensor([0.])
            # self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.detach().numpy()[0], log_probs.detach().numpy(), value.cpu().detach().numpy()
        else:
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            probs = torch.squeeze(dist.log_prob(action)).detach()
            action = torch.squeeze(action).detach().numpy()[0]
            value = torch.squeeze(value).detach()
            return action, probs, value

    def update(self):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = self.memory.sample()
        values = vals_arr[:]
        ### compute advantage ###
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                td_ = reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k]
                a_t += discount * td_
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage).float().to(self.device)
        ### SGD ###
        values = torch.tensor(values).float().to(self.device)
        # state_arr, action_arr, old_prob_arr, advantage, values
        old_prob_arr = torch.tensor(old_prob_arr).float().to(self.device)
        action_arr = torch.tensor(action_arr).float().to(self.device)
        state_arr = torch.tensor(state_arr).float().to(self.device)
        d_set = memDataset(state_arr, action_arr, old_prob_arr, advantage, values)
        train_loader = DataLoader(
            d_set,
            batch_size=self.sgd_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=self.min_batch_collate_func
        )
        for _ in range(self.k_epochs):
            for state_, action_, old_prob_, adv_, value_ in train_loader:
                states = state_.to(self.device)
                old_probs = old_prob_.detach().to(self.device)
                actions = action_.to(self.device)
                adv_ = adv_.to(self.device)
                value_ = value_.detach().to(self.device)
                
                if self.continuous:
                    mu, sigma = self.actor(states)
                    dist = Normal(self.action_bound * mu, sigma)
                    act = dist.sample()
                    new_probs = dist.log_prob(act)
                else:
                    probs = self.actor(states)
                    dist = Categorical(probs)
                    new_probs = dist.log_prob(actions)
                
                critic_value = self.critic(states)
                # critic_value = torch.squeeze(critic_value)
                prob_ratio = torch.exp(new_probs - old_probs)
                weighted_probs = adv_ * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv_
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = value_ # adv_ + value_
                critic_loss = (critic_value - returns.detach())**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # total_loss.backward()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()  
        
    def save_model(self,fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        actor_checkpoint = os.path.join(fpath, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(fpath, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
        
    def load_model(self,fpath):
        actor_checkpoint = torch.load(os.path.join(fpath, 'ppo_actor.pt'), map_location=self.device)
        critic_checkpoint = torch.load(os.path.join(fpath, 'ppo_critic.pt'), map_location=self.device)
        self.actor.load_state_dict(actor_checkpoint) 
        self.critic.load_state_dict(critic_checkpoint)  

