import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class A2C_2:
    def __init__(self,models,memories,cfg):
        self.n_actions = cfg['n_actions']
        self.gamma = cfg['gamma']
        self.device = torch.device(cfg['device']) 
        self.memory = memories['ACMemories']
        self.ac_net = models['ActorCritic'].to(self.device)
        self.ac_optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=cfg['lr'])
    def sample_action(self,state):
        value, dist = self.ac_net(state)
        value = value.detach().numpy().squeeze(0)[0]
        action = np.random.choice(self.n_actions, p=dist.detach().numpy().squeeze(0)) # p_shape(n_actions,1)
        return action,value,dist
    def predict_action(self,state):
        value, dist = self.ac_net(state)
        value = value.detach().numpy().squeeze(0)[0]
        action = np.random.choice(self.n_actions, p=dist.detach().numpy().squeeze(0)) # p_shape(n_actions,1)
        return action,value,dist
    def update(self,next_state,entropy):
        value_pool,log_prob_pool,reward_pool = self.memory.sample()
        next_value,_ = self.ac_net(next_state)
        returns = np.zeros_like(reward_pool)
        for t in reversed(range(len(reward_pool))):
            next_value = reward_pool[t] + self.gamma * next_value # G(s_{t},a{t}) = r_{t+1} + gamma * V(s_{t+1})
            returns[t] = next_value
        returns = torch.tensor(returns, device=self.device)
        value_pool = torch.tensor(value_pool, device=self.device)
        advantages = returns - value_pool
        log_prob_pool = torch.stack(log_prob_pool)
        actor_loss = (-log_prob_pool * advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy
        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()
        self.memory.clear()
    def save_model(self, path):
        from pathlib import Path
        # create path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.ac_net.state_dict(), f"{path}/a2c_checkpoint.pt")

    def load_model(self, path):
        self.ac_net.load_state_dict(torch.load(f"{path}/a2c_checkpoint.pt"))
        
       