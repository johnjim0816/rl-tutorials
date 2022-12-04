#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-30 00:53:03
LastEditor: JiangJi
LastEditTime: 2022-12-04 14:47:54
Discription: default parameters of A2C
'''
        
class AlgoConfig:
    def __init__(self) -> None:
        self.continuous = False # continuous or discrete action space
        self.gamma = 0.99 # discount factor
        self.actor_lr = 3e-4 # learning rate of actor
        self.critic_lr = 1e-3 # learning rate of critic
        self.actor_hidden_dim = 256 # hidden_dim for actor MLP
        self.critic_hidden_dim = 256 # hidden_dim for critic MLP
        self.entropy_coef = 0.001 # entropy coefficient
        self.update_freq = 20 # update policy every n steps