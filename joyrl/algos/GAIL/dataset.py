"""
Author:      Yi Zhang, Master Student @ idrugLab, School of Biology and Biological Engineering, South China Universty of Technology
Created on:  2022/11/17
"""

import torch
from torch.utils.data import Dataset


class TrajDataset(Dataset):
    def __init__(self, trajectories):
        super(TrajDataset, self).__init__()
        self.states = torch.FloatTensor(trajectories['states']) if not isinstance(trajectories['states'],
                                                                                  torch.Tensor) else trajectories[
            'states']
        self.actions = torch.FloatTensor(trajectories['actions']) if not isinstance(trajectories['actions'],
                                                                                    torch.Tensor) else trajectories[
            'actions']
        self.rewards = torch.FloatTensor(trajectories['rewards']) if not isinstance(trajectories['rewards'],
                                                                                    torch.Tensor) else trajectories[
            'rewards']
        self.terminals = torch.Tensor(trajectories['terminals']) if not isinstance(trajectories['terminals'],
                                                                                   torch.Tensor) else trajectories[
            'terminals']

    def __len__(self):
        return self.terminals.size(0) - 1

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'states':
                return self.states
            elif idx == 'actions':
                return self.actions
            elif idx == 'terminals':
                return self.terminals
        else:
            return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx],
                        next_states=self.states[idx + 1], terminals=self.terminals[idx])
