import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
import numpy as np
from common.memories import ReplayBufferQue, ReplayBuffer
class DistributionalNetwork(nn.Module):
    def __init__(self, n_states, n_actions,n_atoms, Vmin, Vmax):
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

        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions * n_atoms)
        self.register_buffer('supports', torch.arange(Vmin, Vmax + self.delta_z, self.delta_z))
        # self.reset_parameters()
    def dist(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.n_actions, self.n_atoms)
        x = torch.softmax(x, dim=-1)
        return x

    def forward(self, x):
        x = self.dist(x)
        x = torch.sum(x * self.supports, dim=2)
        return x
class Agent:
    def __init__(self,cfg) -> None:
        self.n_actions = cfg.n_actions
        self.n_atoms = cfg.n_atoms
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        self.gamma = cfg.gamma

        self.tau = cfg.tau
        self.device = torch.device(cfg.device)

        self.policy_net = DistributionalNetwork(cfg.n_states, cfg.n_actions, cfg.n_atoms, cfg.Vmin, cfg.Vmax).to(self.device)
        self.target_net= DistributionalNetwork(cfg.n_states, cfg.n_actions, cfg.n_atoms, cfg.Vmin, cfg.Vmax).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer_size) # ReplayBufferQue(cfg.capacity)
        self.sample_count = 0

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
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(dim=1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        # calculate the distribution of the next state
        
        with torch.no_grad():
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
            
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


