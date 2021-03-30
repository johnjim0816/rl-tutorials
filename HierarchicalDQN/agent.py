#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-24 22:18:18
LastEditor: John
LastEditTime: 2021-03-30 19:38:15
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import numpy as np
import random,math
from HierarchicalDQN.model import MLP
from common.memory import ReplayBuffer
import torch.optim as optim
class HierarchicalDQN:
    def __init__(self,state_dim,action_dim,cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.frame_idx = 0 
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end ) * math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.policy_net = MLP(2*state_dim, action_dim,cfg.hidden_dim).to(self.device)
        self.meta_policy_net = MLP(state_dim, state_dim,cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=cfg.lr)
        self.meta_optimizer = optim.Adam(self.meta_policy_net.parameters(),lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.meta_memory = ReplayBuffer(cfg.memory_capacity)
    def to_onehot(self,x):
        oh = np.zeros(self.state_dim)
        oh[x - 1] = 1.
        return oh
    def set_goal(self,state):
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                goal = self.meta_policy_net(state).max(1)[1].item() 
        else:
            goal = random.randrange(self.state_dim)
        return goal
    def choose_action(self,state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_value = self.policy_net(state)
                action = q_value.max(1)[1].item()  
        else:
            action = random.randrange(self.action_dim)
        return action
    def update(self):
        if self.batch_size > len(self.memory):
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device,dtype=torch.int64).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch).unsqueeze(1)
        next_state_values = self.policy_net(next_state_batch).max(1)[0].detach()
        # next_state_values = self.target_net(next_state_batch).max(1)[0].detach()  
        expected_q_values = reward_batch + self.gamma * next_state_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values) 
        self.optimizer.zero_grad() 
        loss.backward()
        # for param in self.policy_net.parameters(): 
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 
        # update params of meta_net
        if self.batch_size > len(self.meta_memory):
            return
        meta_state_batch, meta_action_batch, meta_reward_batch, next_meta_state_batch, meta_done_batch = self.meta_memory.sample(self.batch_size)
        meta_state_batch = torch.tensor(meta_state_batch, device=self.device, dtype=torch.float)
        meta_action_batch = torch.tensor(meta_action_batch, device=self.device,dtype=torch.int64).unsqueeze(1)  
        meta_reward_batch = torch.tensor(meta_reward_batch, device=self.device, dtype=torch.float)  
        next_meta_state_batch = torch.tensor(next_meta_state_batch, device=self.device, dtype=torch.float)
        meta_done_batch = torch.tensor(np.float32(meta_done_batch), device=self.device) 
        meta_q_values = self.meta_policy_net(meta_state_batch).gather(dim=1, index=meta_action_batch).unsqueeze(1) 
        next_meta_state_values = self.meta_policy_net(next_meta_state_batch).max(1)[0].detach()
        # next_meta_state_values = self.meta_target_net(next_meta_state_batch).max(1)[0].detach()  
        expected_meta_q_values = meta_reward_batch + self.gamma * next_meta_state_values * (1-meta_done_batch)
        meta_loss = nn.MSELoss()(meta_q_values, expected_meta_q_values) 
        self.meta_optimizer.zero_grad() 
        meta_loss.backward()
        # for param in self.meta_policy_net.parameters(): 
        #     param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step() 
        
        