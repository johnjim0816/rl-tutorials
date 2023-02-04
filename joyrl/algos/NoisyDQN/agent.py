import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
from common.layers import ValueNetwork
from common.memories import ReplayBuffer

'''
This NoisyLinear is modified from the original code from 
https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
''' 

class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(torch.tensor(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(torch.tensor(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class NoisyMLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        super(NoisyMLP, self).__init__()
        self.fc1 =  nn.Linear(input_dim, hidden_dim)
        self.noisy_fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_fc3 = NoisyLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy_fc2(x))
        x = self.noisy_fc3(x)
        return x

    def reset_noise(self):
        self.noisy_fc2.reset_noise()
        self.noisy_fc3.reset_noise()

class Agent:
    def __init__(self, cfg) -> None:
        self.n_actions = cfg.n_actions  
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update

        self.device = torch.device(cfg.device) 

        self.policy_net = NoisyMLP(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = NoisyMLP(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(self.device)
        ## copy parameters from policy net to target net
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer_size)

        self.update_flag = False
    def sample_action(self, state):
        ''' sample action with e-greedy policy 
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        q_value = self.policy_net(state)
        action  = q_value.max(1)[1].item()
        return action
    def update(self):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True
        # beta = min(1.0, self.beta_start + self.sample_count * (1.0 - self.beta_start) / self.beta_frames)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights_batch, indices = self.memory.sample(self.batch_size, beta) 
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) 
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float).unsqueeze(1)
        # weights_batch = torch.tensor(weights_batch, device=self.device, dtype=torch.float)
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)  # shape same to  
        # backpropagation
        self.optimizer.zero_grad()  
        loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 

        if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())   

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)