import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class A2C_2:
    def __init__(self,models,memories,cfg):
        self.n_actions = cfg['n_actions']
        self.device = torch.device(cfg['device']) 
        self.memory = memories['ACMemories']
        self.ac_net = models['ActorCritic']
        self.ac_optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=cfg['lr'])
    def sample_action(self,state):
        value, policy_dist = self.ac_net(state)
        value = value.detach().numpy()[0,0]
        dist = policy_dist.detach().numpy() 
        action = np.random.choice(self.n_actions, p=np.squeeze(dist))
        return action
    def predict_action(self,state):
        pass
    def update(self,next_state,entropy):
        value_pool,reward_pool,log_prob_pool, done_pool = self.memory.sample()
        next_value,_ = self.ac_net(next_state)
        pass
        
       