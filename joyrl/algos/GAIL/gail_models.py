"""
Author:      Yi Zhang, Master Student @ idrugLab, School of Biology and Biological Engineering, South China Universty of Technology
Created on:  2022/12/04
"""
import torch
import torch.nn as nn
from .utils import concat_state_action


class GAILDiscriminator(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.action_size = action_size
        input_layer = nn.Linear(state_size + action_size, hidden_size)
        self.discriminator = nn.Sequential(input_layer, nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                           nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, state, action):
        return self.discriminator(concat_state_action(state, action, self.action_size)).squeeze(dim=1)

    def predict_reward(self, state, action):
        D = self.forward(state, action)
        h = torch.log(D) - torch.log1p(-D)
        return h
