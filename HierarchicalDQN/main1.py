#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-29 10:37:32
LastEditor: John
LastEditTime: 2021-03-30 20:23:55
Discription: 
Environment: 
'''
import math
import random
from memory import ReplayBuffer

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class HierarchicalDQNConfig:
    def __init__(self):
        self.algo = "H-DQN" # name of algo
        self.gamma = 0.99
        self.epsilon_start = 1 # start epsilon of e-greedy policy
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001 # learning rate
        self.memory_capacity = 10000 # Replay Memory capacity
        self.batch_size = 32
        self.train_eps = 3000 # 训练的episode数目
        self.target_update = 2 # target net的更新频率
        self.eval_eps = 20 # 测试的episode数目
        self.eval_steps = 200 # 测试每个episode的最大长度
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测gpu
        self.hidden_dim = 256 # dimension of hidden layer

class Net(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim=256):
        super(Net, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.output_dim = output_dim
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.forward(state).max(1)[1].item()
            return action
        else:
            return random.randrange(self.output_dim)

import gym
env = gym.make('CartPole-v0')
env.seed(1) # 设置env随机种子
num_goals = env.observation_space.shape[0]
num_actions = env.action_space.n
model        = Net(2*num_goals, num_actions)
target_model = Net(2*num_goals, num_actions)

meta_model        = Net(num_goals, num_goals)
target_meta_model = Net(num_goals, num_goals)


optimizer      = optim.Adam(model.parameters())
meta_optimizer = optim.Adam(meta_model.parameters())

replay_buffer      = ReplayBuffer(10000)
meta_replay_buffer = ReplayBuffer(10000)

def to_onehot(x):
    oh = np.zeros(4)
    oh[x - 1] = 1.
    return oh

def update(model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
    state_batch = torch.tensor(state_batch,dtype=torch.float)
    action_batch = torch.tensor(action_batch,dtype=torch.int64).unsqueeze(1)  
    reward_batch = torch.tensor(reward_batch,dtype=torch.float)  
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float)
    done_batch = torch.tensor(np.float32(done_batch))
    q_values = model(state_batch).gather(dim=1, index=action_batch).squeeze(1)
    next_state_values = model(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + 0.99 * next_state_values * (1-done_batch)
    # loss = nn.MSELoss()(q_values, Variable(expected_q_values.data)) 
    loss = nn.MSELoss()(q_values, expected_q_values) 
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step() 


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

frame_idx  = 1
for i in range(300):
    state = env.reset()
    done  = False
    ep_reward = 0
    while not done:
        goal = meta_model.act(state, epsilon_by_frame(frame_idx))
        onehot_goal  = to_onehot(goal)
        meta_state = state
        extrinsic_reward = 0
        while not done and goal != np.argmax(state):
            goal_state  = np.concatenate([state, onehot_goal])
            action = model.act(goal_state, epsilon_by_frame(frame_idx))
            next_state, reward, done, _ = env.step(action)
            ep_reward   += reward
            extrinsic_reward += reward
            intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0
            replay_buffer.push(goal_state, action, intrinsic_reward, np.concatenate([next_state, onehot_goal]), done)
            state = next_state
            frame_idx  += 1
            update(model, optimizer, replay_buffer, 32)
            update(meta_model, meta_optimizer, meta_replay_buffer, 32)
    meta_replay_buffer.push(meta_state, goal, extrinsic_reward, state, done)
    print(i+1,ep_reward)

