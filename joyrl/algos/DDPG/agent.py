#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2022-12-06 22:50:45
@Discription: 
@Environment: python 3.7.7
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.memories import ReplayBufferQue
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class OUNoise(object):
    '''Ornstein–Uhlenbeck Noise'''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu # mean of noise
        self.theta        = theta # theta of noise
        self.sigma        = max_sigma # sigma of noise
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.n_actions   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu # reset the noise
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions) # Ornstein–Uhlenbeck process
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # linearly decay the action randomness
        return np.clip(action + ou_obs, self.low, self.high) # add noise to action

class Agent:
    def __init__(self,cfg):
        self.n_states = cfg.n_states
        self.n_actions = cfg.n_actions
        self.action_space = cfg.action_space # action space
        self.ou_noise = OUNoise(self.action_space) # initialize the noise
        self.batch_size = cfg.batch_size # batch size for updating actor and critic
        self.gamma = cfg.gamma # reward discount factor
        self.tau = cfg.tau # soft update parameter
        self.sample_count = 0 # record the number of sample action from the environment
        self.update_flag = False # whether to update the target network
        self.device = torch.device(cfg.device)
        self.critic = Critic(self.n_states,self.n_actions,hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.target_critic = Critic(self.n_states,self.n_actions,hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor = Actor(self.n_states,self.n_actions,hidden_dim=cfg.actor_hidden_dim).to(self.device)
        self.target_actor = Actor(self.n_states,self.n_actions,hidden_dim=cfg.actor_hidden_dim).to(self.device).to(self.device)
        # copy weights from critic to target_critic (hard update)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        # copy weights from actor to target_actor (hard update)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),  lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBufferQue(cfg.buffer_size)

    def sample_action(self, state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        action_tanh = self.actor(state) # action_tanh is in [-1, 1]
        # convert action_tanh to action in the original action space
        action_scale = torch.FloatTensor((self.action_space.high - self.action_space.low) / 2.).to(self.device)
        action_bias = torch.FloatTensor((self.action_space.high + self.action_space.low) / 2.).to(self.device)
        action = action_scale * action_tanh + action_bias
        action = action.cpu().detach().numpy()[0]
        # add noise to action
        action = self.ou_noise.get_action(action, self.sample_count) 
        return action
    @torch.no_grad()
    def predict_action(self, state):
        ''' predict action
        '''
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        action_tanh = self.actor(state) # action_tanh is in [-1, 1]
        # convert action_tanh to action in the original action space
        action_scale = torch.FloatTensor((self.action_space.high - self.action_space.low) / 2.).to(self.device)
        action_bias = torch.FloatTensor((self.action_space.high + self.action_space.low) / 2.).to(self.device)
        action = action_scale * action_tanh + action_bias
        action = action.cpu().detach().numpy()[0]
        return action

    def update(self):
        if len(self.memory) < self.batch_size: # when memory size is less than batch size, return
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True
        # sample a random minibatch of N transitions from R
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # convert to tensor
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
       
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # soft update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
    def save_model(self,fpath):
        from pathlib import Path
        # create path
        
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{fpath}/actor_checkpoint.pt")

    def load_model(self,fpath):
        actor_ckpt = torch.load(f"{fpath}/actor_checkpoint.pt", map_location=self.device)
        self.actor.load_state_dict(actor_ckpt) 