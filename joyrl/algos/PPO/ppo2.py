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
from torch.distributions import Categorical,Normal
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

class memDataset(Dataset):
    def __init__(self, states, actions, old_log_probs, advantage, td_target):
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
        self.batch_size = cfg.sgd_batch_size
        self.gae_lambda = cfg.gae_lambda
        self.device = torch.device(cfg.device) 
        self.actor = models['Actor'].to(self.device)
        self.critic = models['Critic'].to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = memory
        self.loss = 0
        
        self.actor_nums = cfg.actor_nums
        self.update_count = 0
        self.max_steps = cfg.max_steps

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if self.continuous:
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu, sigma)
            action = dist.sample()
            # value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            # value = value.detach().cpu().numpy().squeeze(0).item() # detach() to avoid gradient
            # log_probs = dist.log_prob(action).item() # Tensor([0.])
            # self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.cpu().detach().numpy() # ,log_probs,value
        else:
            probs = self.actor(state)
            dist = Categorical(probs)
            value = self.critic(state)
            action = dist.sample()
            # probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            # value = torch.squeeze(value).item()
            return action #, probs, value

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if self.continuous:
            mu, sigma = self.actor(state)
            dist = Normal(self.action_bound * mu.view(1,), sigma.view(1,))
            action = dist.sample()
            # value = self.critic(state)
            # self.entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
            # value = value.detach().cpu().numpy().squeeze(0)[0] # detach() to avoid gradient
            # log_probs = dist.log_prob(action).item() # Tensor([0.])
            # self.entropy = dist.entropy().cpu().detach().numpy().squeeze(0) # detach() to avoid gradient
            return action.cpu().numpy() #,log_probs,value.cpu()
        else:
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            # probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            # value = torch.squeeze(value).item()
            return action #, probs, value
        
    @staticmethod
    def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        adv = 0
        adv_list = []
        for td in td_delta[::-1]:
            adv = gamma * lmbda * adv + td
            adv_list.append(adv)
        adv_list.reverse()
        return torch.FloatTensor(adv_list)

    def update(self):
        """PPO n-actors on epsiode
        for actor=1, 2, ..., N do
            run Policy pi_{theta_{old}} in environment for max_steps
            compute advantage
        
        optimize L, with k_epochs and minibatch_size M <= N * max_steps 
        """
        self.update_count += 1
        if self.update_count % self.actor_nums:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample()
        # to tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        rewards = (rewards + 10.0) / 10.0
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).view(-1, 1)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.gae_lambda, td_delta)
        
        # old log probs
        mu, std = self.actor(states)
        action_dist = Normal(self.action_bound * mu.detach(), std.detach())
        old_log_probs = action_dist.log_prob(actions)
        
        # datdLoader
        d_set = memDataset(states, actions, old_log_probs, advantage, td_target)
        train_loader = DataLoader(
            d_set,
            batch_size=min(self.batch_size, self.max_steps * self.actor_nums),
            shuffle=True,
            drop_last=True
        )
        ### SGD ###
        for _ in range(self.k_epochs):
            for state, action, old_log_prob, adv, td_v in train_loader:
                state = state.to(self.device)
                old_log_prob = old_log_prob.to(self.device)
                action = action.to(self.device)
                adv = adv.to(self.device)
                td_v = td_v.to(self.device)
                
                # compute actor loss
                mu, std = self.actor(state)
                action_dist = Normal(self.action_bound * mu, std)
                new_log_prob = action_dist.log_prob(action)
                # e^(log(a) - log(b)) = e^log(a) / e^log(b) = a / b
                prob_ratio = torch.exp(new_log_prob - old_log_prob)
                weighted_probs = prob_ratio * adv 
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # compute critic loss
                critic_loss = torch.mean(
                    F.mse_loss(self.critic(state), td_v.detach())
                )
                
                # backwards
                total_loss = actor_loss + 0.5 * critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()  

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        actor_checkpoint = os.path.join(fpath, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(fpath, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
        
    def load_model(self, fpath):
        actor_checkpoint = torch.load(os.path.join(fpath, 'ppo_actor.pt'), map_location=self.device)
        critic_checkpoint = torch.load(os.path.join(fpath, 'ppo_critic.pt'), map_location=self.device)
        self.actor.load_state_dict(actor_checkpoint) 
        self.critic.load_state_dict(critic_checkpoint)  


