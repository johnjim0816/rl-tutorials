#!/usr/bin/env python
# coding=utf-8
'''
Author: GuoShiCheng
Email: guoshichenng@gmail.com
Date: 2022-11-19 09:56:33
LastEditor: GuoShiCheng
LastEditTime: 2022-11-19 09:56:33
Discription: theAlley,walkInThePark
Environment: python 3.7.7
'''

'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2022-10-31 00:07:19
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

class VI:
    def __init__(self, cfg):
        self.Q_table = np.zeros(cfg.n_states)
        # The action selected by the policy must be of type int
        self.policy = np.zeros(cfg.n_states, dtype=int)
        self.n_actions = cfg.n_actions  
        self.n_states = cfg.n_states  
        self.P = cfg.env_P
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma
        self.update_flag = False 
        self.max_Q_value = 1e50
        # self.threshold = 1e-5

    def sample_action(self, state):
        '''sample action
        '''
        action = self.policy[state]
        return action

    def predict_action(self,state):
        ''' predict action
        '''
        action = self.policy[state]
        return action

    def update(self):
        '''
        Iterative policy and Q_table
        '''
        # Prevent Q_table to infinity
        if self.Q_table[0] > self.max_Q_value:
            pass
        else:
            updated_value_table = np.copy(self.Q_table)
            for state in range(self.n_states):
                Q_value = []
                for action in range(self.n_actions):
                    next_states_rewards = []
                    for next_sr in self.P[state][action]:
                        trans_prob, next_state, reward_prob, _ = next_sr
                        next_states_rewards.append((trans_prob * (reward_prob + self.gamma * updated_value_table[next_state])))
                    Q_value.append(np.sum(next_states_rewards))
                # 1.Update Q_table
                self.Q_table[state] = max(Q_value)
                # 2.Policy Improvement
                self.policy[state] = np.argmax(Q_value)

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy, f"{fpath}checkpoint.pt")

    def load_model(self, fpath):
        self.policy = torch.load(f"{fpath}checkpoint.pt")
        

if __name__=="__main__":
    agent = VI()