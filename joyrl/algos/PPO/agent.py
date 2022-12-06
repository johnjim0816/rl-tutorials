import os

import torch
from torch.distributions import Categorical,Normal
import numpy as np
from common.models import ActorSoftmax, ActorNormal, Critic
from common.memories import PGReplay

class Agent:
    def __init__(self,cfg) -> None:
        
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.continuous = cfg.continuous # continuous action space
        self.action_space = cfg.action_space
        if self.continuous:
            self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            self.actor = ActorNormal(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        else:
            self.actor = ActorSoftmax(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states,1,hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.eps_clip = cfg.eps_clip # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sample_count = 0
        self.update_freq = cfg.update_freq

    def sample_action(self,state):
        self.sample_count += 1
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.actor(state)
            mean = mu * self.action_scale + self.action_bias
            std = sigma
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32), torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32))
            self.log_probs = dist.log_prob(action).detach()
            return action.detach().cpu().numpy()[0]
        else: 
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            probs = self.actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            self.log_probs = dist.log_prob(action).detach()
            return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self,state):
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.actor(state)
            mean = mu * self.action_scale + self.action_bias
            std = sigma
            dist = Normal(mean, std)
            action = dist.sample()
            self.log_probs = dist.log_prob(action).detach()
            return action.detach().cpu().numpy()[0]
        else: 
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            probs = self.actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            self.log_probs = dist.log_prob(action).detach()
            return action.detach().cpu().numpy().item()
    def update(self):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for _ in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            if self.continuous:
                mu, sigma = self.actor(old_states)
                mean = mu * self.action_scale + self.action_bias
                std = sigma
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(old_actions)
            else:
                probs = self.actor(old_states)
                dist = Categorical(probs)
                # get new action probabilities
                new_log_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_log_probs - old_log_probs) # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = - (torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean())
            # compute critic loss
            critic_loss = (values - returns).pow(2).mean()
            tot_loss = actor_loss + 0.5 * critic_loss
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            tot_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), fpath + 'actor.pth')
        torch.save(self.critic.state_dict(), fpath + 'critic.pth')
    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/actor.pth", map_location=self.device)
        critic_ckpt = torch.load(f"{fpath}/critic.pth", map_location=self.device)
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)
    def save_traj(self, traj, fpath):
        from pathlib import Path
        import pickle
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        traj_pkl = os.path.join(fpath, 'traj.pkl')
        with open(traj_pkl, 'wb') as f:
            pickle.dump(traj, f)