#!/usr/bin/env python
# coding=utf-8
'''
Author: GuoShiCheng
Email: guoshichenng@gmail.com
Date: 2022-12-05 16:51:42
LastEditor: GuoShiCheng
LastEditTime: 2022-12-07 22:06:18
Discription: Mario
Environment: python 3.7.7
'''
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2022-11-19 17:20:02
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from common.layers import ValueNetwork
from common.memories import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class Agent:
    def __init__(self,cfg):
        self.n_actions = cfg.n_actions
        self.n_states = cfg.n_states
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.policy_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_states, self.n_actions).to(self.device)
        ## copy parameters from policy net to target net
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        # self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        self.memory = ReplayBuffer(cfg.buffer_size)
        self.update_flag = False 
        
    def sample_action(self, state):
        ''' sample action with e-greedy policy
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action
    # @torch.no_grad()
    # def sample_action(self, state):
    #     ''' sample action with e-greedy policy
    #     '''
    #     self.sample_count += 1
    #     # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
    #     self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
    #         math.exp(-1. * self.sample_count / self.epsilon_decay) 
    #     if random.random() > self.epsilon:
    #         state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
    #         q_values = self.policy_net(state)
    #         action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
    #     else:
    #         action = random.randrange(self.n_actions)
    #     return action
    def predict_action(self,state):
        ''' predict action
        '''
        with torch.no_grad():
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action
    def update(self):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True
        # sample a batch of transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
        # print(state_batch.shape,action_batch.shape,reward_batch.shape,next_state_batch.shape,done_batch.shape)
        # compute current Q(s_t,a), it is 'y_j' in pseucodes
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        # print(q_values.requires_grad)
        # compute max(Q(s_t+1,A_t+1)) respects to actions A, next_max_q_value comes from another net and is just regarded as constant for q update formula below, thus should detach to requires_grad=False
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        # print(q_values.shape,next_q_values.shape)
        # compute expected q value, for terminal state, done_batch[0]=1, and expected_q_value=rewardcorrespondingly
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        # print(expected_q_value_batch.shape,expected_q_value_batch.requires_grad)
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)  # shape same to  
        # backpropagation
        self.optimizer.zero_grad()  
        loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 
        if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())   

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
