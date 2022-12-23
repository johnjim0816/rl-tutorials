#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-12-13 13:48:59
LastEditor: JiangJi
LastEditTime: 2022-12-23 17:44:39
Discription: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
import numpy as np
from collections import deque
class LSTM(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim = 64):
        super(LSTM, self).__init__()
        self.l1 = nn.Linear(n_states, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True) # batch_first=True means input and output shape is (batch_size, seq_len, features)
        self.l2 = nn.Linear(hidden_dim, n_actions)
    def forward(self, x, h, c):
        x = F.relu(self.l1(x))
        x, (h, c) = self.lstm(x, (h, c))
        x = self.l2(x)
        return x, (h, c)
class GRUMemory:
    def __init__(self, capacity: int, lookup_size = 2) -> None:
        self.capacity = capacity # capacity of memory
        self.lookup_size = lookup_size # lookup size for sequential sampling
        self.buffer = deque(maxlen=self.capacity)
        self.lookup_buffer = []
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        if len(self.lookup_buffer)==0:
            self.lookup_buffer.append(np.array(transitions))
            self.lookup_buffer = np.array(self.lookup_buffer)
        elif len(self.lookup_buffer) < self.lookup_size:
            self.lookup_buffer = np.concatenate((self.lookup_buffer, np.array(transitions)[np.newaxis, :]), axis=0)
        if len(self.lookup_buffer) == self.lookup_size:
            self.lookup_buffer = self.lookup_buffer.T
            print('lookup_buffer is full, push to buffer')
            print(self.lookup_buffer)
            self.buffer.append(self.lookup_buffer)
            self.lookup_buffer = []
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)
class Agent:
    def __init__(self,cfg) -> None:
        self.sample_count = 0
        self.device = torch.device(cfg.device)
        self.policy_net = LSTM(cfg.n_states, cfg.n_actions)
        self.target_net = LSTM(cfg.n_states, cfg.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
    def sample_action(self, state):
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values, (h,c) = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action
    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        q_values, _ = self.policy_net(state)
        action = q_values.max(1)[1].item()
        return action
    def update(self):
        pass
