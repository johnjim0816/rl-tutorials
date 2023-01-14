import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
import numpy as np
from common.memories import ReplayBufferQue, ReplayBuffer, ReplayTree

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

class DistributionalNetwork(nn.Module):
    def __init__(self, n_states, hidden_dim, n_actions,n_atoms, Vmin, Vmax):
        super(DistributionalNetwork, self).__init__()
        self.n_atoms = n_atoms  # number of atoms
        '''Vmin,Vmax: Range of the support of rewards. Ideally, it should be [min, max], '
                             'where min and max are referred to the min/max cumulative discounted '
                             'reward obtainable in one episode. Defaults to [0, 200].'
        '''
        self.Vmin = Vmin # minimum value of support
        self.Vmax = Vmax # maximum value of support
        self.delta_z = (Vmax - Vmin) / (n_atoms - 1)
        self.n_actions = n_actions

        self.fc1 =  nn.Linear(n_states, hidden_dim)
        self.noisy_value2 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_value3 = NoisyLinear(hidden_dim, n_atoms)

        self.noisy_advantage2 = NoisyLinear(hidden_dim, hidden_dim) # NoisyDQN + Dueling DQN
        self.noisy_advantage3 = NoisyLinear(hidden_dim, n_actions * n_atoms)

        self.register_buffer('supports', torch.arange(Vmin, Vmax + self.delta_z, self.delta_z))
        # self.reset_parameters()
    def dist(self, x):
        x = torch.relu(self.fc1(x))

        value = F.relu(self.noisy_value2(x))
        value = self.noisy_value3(value).view(-1, 1, self.n_atoms)

        advantage = F.relu(self.noisy_advantage2(x))
        advantage = self.noisy_advantage3(advantage).view(-1, self.n_actions, self.n_atoms)

        x = value + advantage - advantage.mean(dim=1, keepdim=True)
        x = x.view(-1, self.n_actions, self.n_atoms)
        x = torch.softmax(x, dim=-1)
        return x

    def forward(self, x):
        x = self.dist(x)
        x = torch.sum(x * self.supports, dim=2)
        return x

    def reset_noise(self):
        self.noisy_value2.reset_noise()
        self.noisy_value3.reset_noise()

        self.noisy_advantage2.reset_noise()
        self.noisy_advantage3.reset_noise()  

class Agent:
    def __init__(self,cfg) -> None:
        self.n_actions = cfg.n_actions
        self.n_atoms = cfg.n_atoms
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.device = torch.device(cfg.device)

        self.policy_net = DistributionalNetwork(cfg.n_states, cfg.hidden_dim, cfg.n_actions, cfg.n_atoms, cfg.Vmin, cfg.Vmax).to(self.device)
        self.target_net= DistributionalNetwork(cfg.n_states, cfg.hidden_dim, cfg.n_actions, cfg.n_atoms, cfg.Vmin, cfg.Vmax).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        # self.memory = ReplayBuffer(cfg.buffer_size) # ReplayBufferQue(cfg.capacity)
        self.memory = ReplayTree(cfg.buffer_size)
        self.sample_count = 0

        self.n_step = cfg.n_step ## used for N-step DQN 

        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update

    def sample_action(self, state):
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            action = self.predict_action(state)
        else:
            action = random.randrange(self.n_actions)
        return action

    def predict_action(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            # print ("state", state)
            q_values = self.policy_net(state)
            action  = q_values.max(1)[1].item()
            # action = q_values.argmax() // self.n_atoms
            # action = action.item()  # choose action corresponding to the maximum q value
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        (states, actions, rewards, next_states, dones), idxs_batch, is_weights_batch = self.memory.sample(
            self.batch_size)

        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(dim=1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        # calculate the distribution of the next state
        
        with torch.no_grad():
            # next_action = self.policy_net --> DDQN  self.target_net --> DQN
            next_action = self.policy_net(next_states).detach().max(1)[1].unsqueeze(dim=1).unsqueeze(dim=1).expand(self.batch_size, 1, self.n_atoms)
            next_dist = self.target_net.dist(next_states).detach()
            next_dist = next_dist.gather(1, next_action).squeeze(dim=1)

            # calculate the distribution of the current state
            Tz = rewards + (1 - dones) * self.gamma * self.target_net.supports
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            b = (Tz - self.Vmin) / self.policy_net.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).unsqueeze(dim=1).expand(self.batch_size, self.n_atoms).to(self.device)
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, torch.tensor(l + offset,dtype=torch.int).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, torch.tensor(u + offset,dtype=torch.int).view(-1), (next_dist * (b - l.float())).view(-1))
        # calculate the loss
        dist = self.policy_net.dist(states)
        actions = actions.unsqueeze(dim=1).expand(self.batch_size, 1, self.n_atoms)
        dist = dist.gather(1, actions).squeeze(dim=1)
        loss = -(proj_dist * dist.log()).sum(1).mean()
        
        ## update the weight in the PER DQN 
        q_value_batch = torch.sum(proj_dist * self.target_net.supports, dim=1).unsqueeze(dim=1)
        expected_q_value_batch = torch.sum(dist * self.target_net.supports, dim=1) .unsqueeze(dim=1)
        abs_errors = np.sum(np.abs(q_value_batch.cpu().detach().numpy() - expected_q_value_batch.cpu().detach().numpy()), axis=1)
        self.memory.batch_update(idxs_batch, abs_errors) 

        # update the network
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # soft update the target network
        if self.sample_count % self.target_update == 0:
            if self.tau == 1.0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

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


