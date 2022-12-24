#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-12-24 20:41:56
LastEditor: JiangJi
LastEditTime: 2022-12-24 20:53:52
Discription: 
'''
import torch
class AlgoConfig:
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.tau = 1.0 # 1.0 means hard update
        self.v_min = -10 # support of C51  
        self.v_max = 10 # support of C51 
        self.num_atoms = 51 # support of C51
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms) # support of C51
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1) # support of C51

        self.batch_size = 32 # batch size
        self.lr = 0.0001 # learning rate
        self.target_update = 800 # target network update frequency
        self.memory_capacity = 10000 # size of replay buffer
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate