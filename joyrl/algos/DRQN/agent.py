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
import copy
from collections import deque
class LSTM(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim = 64):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.l1 = nn.Linear(n_states, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True) # 
        self.l2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, h, c):
        x = F.relu(self.l1(x))
        x, (h, c) = self.lstm(x, (h, c))
        x = self.l2(x)
        return x, h, c

    def sample_action(self, state, h, c, epsilon):
        output = self.forward(state, h,c)

        if random.random() < epsilon:
            return random.randint(0,1), output[1], output[2]
        else:
            return output[0].argmax().item(), output[1] , output[2]

    def init_hidden_state(self, batch_size, training=None):
        if training is True:
            return torch.zeros([1, batch_size, self.hidden_dim]), torch.zeros([1, batch_size, self.hidden_dim])
        else:
            return torch.zeros([1, 1, self.hidden_dim]), torch.zeros([1, 1, self.hidden_dim])
        
class GRUMemory:
    def __init__(self, capacity: int, max_epi_num:int, max_epi_len:int, lookup_size = 2) -> None:
        self.capacity = capacity # capacity of memory
        self.lookup_size = lookup_size # lookup size for sequential sampling
        self.buffer = deque(maxlen=max_epi_num)
        self.lookup_buffer = []
        self.max_epi_len = max_epi_len
    def push(self, episode):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(episode)
    def sample(self, batch_size: int, sequential: bool = False):

        sampled_buffer = []
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            idx = np.random.randint(0, len(self.buffer))
            sampled_buffer.append(self.buffer[idx].sample(len(self.buffer[idx])))
            return sampled_buffer # zip(*sampled_buffer)
        else:
            sampled_episodes = random.sample(self.buffer, batch_size)
            min_step = self.max_epi_len
            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.lookup_size: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_size+1)
                    sample = copy.deepcopy(episode.buffer[idx:idx+self.lookup_size])
                    sampled_buffer.append(sample)
                else:
                    # print ("episode = ", episode.buffer)
                    idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                    sample = copy.deepcopy(episode.buffer[idx:idx+min_step])
                    sampled_buffer.append(sample)                            
            return sampled_buffer
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)
class Agent:
    def __init__(self,cfg) -> None:
        self.sample_count = 0
        self.device = torch.device(cfg.device)
        self.gamma = cfg.gamma  

        self.policy_net = LSTM(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(self.device)
        self.target_net = LSTM(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

        self.memory = GRUMemory(cfg.buffer_size, max_epi_num=cfg.max_epi_num, max_epi_len=cfg.max_epi_len, lookup_size=cfg.lookup_step) 
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay

        self.epsilon = cfg.epsilon_start
        self.batch_size = cfg.batch_size
        self.min_epi_num = cfg.min_epi_num
        self.hidden_dim = cfg.hidden_dim

        self.update_flag = False
        self.target_update = cfg.target_update

    def sample_action(self, state, h, c):
        
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        # self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #     math.exp(-1. * self.sample_count / self.epsilon_decay)

        action, h, c = self.policy_net.sample_action(torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device), 
                                              h.to(self.device), c.to(self.device), self.epsilon) 

        return action, h, c
    @torch.no_grad()
    def predict_action(self, state, h, c):
        output = self.policy_net.forward(torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device), \
            h.to(self.device), c.to(self.device))
        return output[0].argmax().item(), output[1] , output[2] 

    def update(self):
        if len(self.memory) < self.min_epi_num :
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True

        episode_batch = self.memory.sample(self.batch_size)
        state_batch = [] ; action_batch = [] ; reward_batch = [] ; next_state_batch = [] ; done_batch = []
        for i in range(self.batch_size):
            cur_state = [trans[0] for trans in episode_batch[i] ] ; state_batch.append(cur_state)
            cur_action = [trans[1] for trans in episode_batch[i] ] ; action_batch.append(cur_action)
            cur_reward = [trans[2] for trans in episode_batch[i] ] ; reward_batch.append(cur_reward)
            cur_next_state = [trans[3] for trans in episode_batch[i] ] ; next_state_batch.append(cur_next_state)
            cur_done = [trans[4] for trans in episode_batch[i] ] ; done_batch.append(cur_done)
        
        state_batch = np.array(state_batch) ; action_batch = np.array(action_batch) ; reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch) ; done_batch = np.array(done_batch)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(2)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(2)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)  # shape(batchsize,n_states)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float).unsqueeze(2)

        h_target, c_target = self.target_net.init_hidden_state(batch_size=self.batch_size, training=True) ## should be changed
        h_target = h_target.to(self.device) ; c_target = c_target.to(self.device) 
        next_max_q_value_batch, _, _ = self.target_net(next_state_batch, h_target, c_target)
        next_max_q_value_batch = next_max_q_value_batch.max(2)[0].detach().unsqueeze(2)
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)

        h_policy, c_policy = self.policy_net.init_hidden_state(batch_size=self.batch_size, training=True) ## should be changed
        h_policy = h_policy.to(self.device) ; c_policy = c_policy.to(self.device) 
        q_value_batch, _, _ = self.policy_net(state_batch, h_policy, c_policy)
        q_value_batch = q_value_batch.gather(dim=2, index=action_batch) # shape(batchsize,1),requires_grad=True

        # loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)  # shape same to  
        loss = F.smooth_l1_loss(q_value_batch, expected_q_value_batch)  # shape same to  
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
        checkpoint = torch.load(f"{fpath}/checkpoint.pt",map_location=self.device)
        self.target_net.load_state_dict(checkpoint)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
