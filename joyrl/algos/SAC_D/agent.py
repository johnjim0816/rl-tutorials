import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from common.memories import ReplayBuffer
import random
import math
import numpy as np

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        xu = state
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class PolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(PolicyNet, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        probs = F.softmax(x, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        return x, probs + z


class Agent:
    def __init__(self,cfg) -> None:
        self.n_states = cfg.n_states
        self.n_actions = cfg.n_actions
        self.action_space = cfg.action_space
        self.sample_count = 0
        self.update_count = 0
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.n_epochs = cfg.n_epochs
        self.target_update = cfg.target_update
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.buffer_size)
        self.device = torch.device(cfg.device) 
        self.critic = QNetwork(cfg.n_states,cfg.n_actions, cfg.hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr)
        self.critic_target = QNetwork(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(self.device)
        
        self.target_entropy = 0.98 * (-np.log(1 / self.n_actions))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr)

        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.policy = PolicyNet(cfg.n_states, cfg.n_actions, cfg.hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=cfg.lr)

    def sample_action(self,state):
        self.sample_count+=1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            q_values, _ = self.policy(state)
            action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action
        
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        q_values, _ = self.policy(state)
        action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action # .detach().cpu().numpy()[0]
    def update(self):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return
        for i in range(self.n_epochs):
            self.update_count += 1
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size=self.batch_size)

            state_batch = torch.tensor(state_batch, device=self.device,  dtype=torch.float)
            action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
            reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1)
            next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
            done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_probs = self.policy(next_state_batch)
                next_log_probs = torch.log(next_probs)

                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
                min_qf_next_target = (next_probs * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
                next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)

            qf1, qf2 = self.critic(state_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1 = qf1.gather(1, action_batch) ; qf2 = qf2.gather(1, action_batch)

            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            for param in self.critic.parameters():  
                param.grad.data.clamp_(-1, 1)
            self.critic_optim.step()


            pi, probs = self.policy(state_batch)
            log_probs = torch.log(probs)
            with torch.no_grad():
                qf1_pi, qf2_pi = self.critic(state_batch)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = (probs * ((self.alpha * log_probs) - min_qf_pi)).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            
            self.policy_optim.zero_grad()
            policy_loss.backward()
            for param in self.policy.parameters():  
                param.grad.data.clamp_(-1, 1)            
            self.policy_optim.step()

            log_probs = (probs * log_probs).sum(-1)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

            # hard update
            if self.update_count % self.target_update == 0:
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_( param.data )

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, f"{fpath}/checkpoint.pt")
        
    
    def load_model(self, fpath):
        checkpoint = torch.load(f"{fpath}/checkpoint.pt", map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])