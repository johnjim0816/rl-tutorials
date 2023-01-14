#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-01-08 16:35:47
LastEditor: JiangJi
LastEditTime: 2023-01-11 13:28:15
Discription: 
'''
class AlgoConfig:
    def __init__(self) -> None:
        self.gamma = 0.99 # discount factor
        self.actor_lr = 3e-4 # learning rate of actor
        self.critic_lr = 1e-3 # learning rate of critic
        self.actor_hidden_dim = 128 # hidden_dim for actor MLP
        self.critic_hidden_dim = 128 # hidden_dim for critic MLP
        self.entropy_coef = 0.05 # entropy coefficient
        self.update_freq = 20 # update policy every n steps
