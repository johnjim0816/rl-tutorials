import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from torch.distributions import Categorical
import gym
import numpy as np

class SoftQ:
    def __init__(self,n_actions,model,memory,cfg):
        self.memory = memory
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma  # discount factor
        self.batch_size = cfg.batch_size
        self.device = torch.device(cfg.device) 
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 复制参数
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.losses = [] # save losses

    def sample_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state)
            v = self.alpha * torch.log(torch.sum(torch.exp(q/self.alpha), dim=1, keepdim=True)).squeeze()
            dist = torch.exp((q-v)/self.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()
    def predict_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state)
            v = self.alpha * torch.log(torch.sum(torch.exp(q/self.alpha), dim=1, keepdim=True)).squeeze()
            dist = torch.exp((q-v)/self.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()
    def update(self):
        if len(self.memory) < self.batch_size: # when the memory capacity does not meet a batch, the network will not update
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device) 
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)
            next_v = self.alpha * torch.log(torch.sum(torch.exp(next_q/self.alpha), dim=1, keepdim=True))
            y = reward_batch + (1 - done_batch ) * self.gamma * next_v
        print()
        loss = F.mse_loss(self.policy_net(state_batch).gather(1, action_batch.long()), y)
        self.losses.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
