"""
Author:      Yi Zhang, Master Student @ idrugLab, School of Biology and Biological Engineering, South China Universty of Technology
Created on:  2022/12/04
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import autograd
from torch.utils.data import DataLoader
from torch.nn import functional as F


def concat_state_action(states, actions, n_action):
    return torch.cat([torch.FloatTensor(states), F.one_hot(torch.tensor(actions, dtype=torch.int64), n_action)], dim=1)

class GAILDiscriminator(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super().__init__()
        self.action_size = n_actions
        input_layer = nn.Linear(n_states + n_actions, hidden_dim)
        self.discriminator = nn.Sequential(input_layer, nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                           nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, state, action):
        return self.discriminator(concat_state_action(state, action, self.action_size)).squeeze(dim=1)

    def predict_reward(self, state, action):
        D = self.forward(state, action)
        h = torch.log(D) - torch.log1p(-D)
        return h
