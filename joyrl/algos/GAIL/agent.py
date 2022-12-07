from collections import deque

import torch
from torch.distributions import Categorical
import numpy as np
from common.models import ActorSoftmax, Critic
from common.memories import PGReplay
import pickle
import os
from torch import optim, autograd
from torch.utils.data import DataLoader
from torch.nn import functional as F
from joyrl.algos.GAIL.dataset import TrajDataset
from joyrl.algos.GAIL.gail_models import GAILDiscriminator


class Agent:
    def __init__(self, cfg) -> None:

        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.actor = ActorSoftmax(cfg.n_states, cfg.n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs  # update policy for K epochs
        self.eps_clip = cfg.eps_clip  # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef  # entropy coefficient
        self.sample_count = 0
        self.update_freq = cfg.update_freq
        if cfg.mode == 'train':
            pkl_path = os.path.join(f"tasks/{cfg.load_path}/traj/", 'traj.pkl')
            with open(pkl_path, 'rb') as handle:
                self.expert_trajectories = TrajDataset(pickle.load(handle))
            self.discriminator = GAILDiscriminator(cfg.n_states,
                                                   cfg.n_actions, cfg.hidden_dim)
            self.discriminator_optimiser = optim.RMSprop(self.discriminator.parameters(), lr=cfg.lr)
            self.policy_trajectory_replay_buffer = deque(maxlen=cfg.imitation_replay_size)

    def sample_action(self, state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()

    def update(self, cfg):
        # states = policy_trajectory_replays['states']
        # actions = policy_trajectory_replays['actions']
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        print("update policy...")
        states, actions, _, _, _ = self.memory.sample()
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        with torch.no_grad():
            old_rewards = self.discriminator.predict_reward(states, actions)
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
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # 1e-5 to avoid division by zero
        for _ in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states)  # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs)  # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()

    def adversarial_update(self, cfg):
        if self.sample_count % self.update_freq != 0:
            return
        print("update adversarial network...")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        policy_trajectory_replays = {'states': old_states, 'actions': old_actions, 'rewards': old_rewards,
                                     'terminals': old_dones}
        policy_trajectory = TrajDataset(policy_trajectory_replays)
        for _ in range(cfg.adversarial_epochs):
            expert_dataloader = DataLoader(self.expert_trajectories, batch_size=cfg.adversarial_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=cfg.num_workers)
            policy_dataloader = DataLoader(policy_trajectory, batch_size=cfg.adversarial_batch_size, shuffle=True,
                                           drop_last=True,
                                           num_workers=cfg.num_workers)

            # Iterate over expert and policy data
            for expert_transition, policy_transition in zip(expert_dataloader, policy_dataloader):
                expert_state, expert_action, expert_next_state, expert_terminal = expert_transition['states'], \
                                                                                  expert_transition['actions'], \
                                                                                  expert_transition['next_states'], \
                                                                                  expert_transition['terminals']
                policy_state, policy_action, policy_next_state, policy_terminal = policy_transition['states'], \
                                                                                  policy_transition['actions'], \
                                                                                  policy_transition['next_states'], \
                                                                                  policy_transition['terminals']

                d_expert = self.discriminator(expert_state, expert_action)
                d_policy = self.discriminator(policy_state, policy_action)

                # Binary logistic regression
                self.discriminator_optimiser.zero_grad()
                expert_loss = F.binary_cross_entropy(d_expert,
                                                     torch.ones_like(d_expert))  # Loss on "real" (expert) data
                autograd.backward(expert_loss, create_graph=True)
                r1_reg = 0
                for param in self.discriminator.parameters():
                    r1_reg += param.grad.norm()  # R1 gradient penalty
                policy_loss = F.binary_cross_entropy(d_policy,
                                                     torch.zeros_like(d_policy))  # Loss on "fake" (policy) data
                (policy_loss + cfg.r1_reg_coeff * r1_reg).backward()
                self.discriminator_optimiser.step()

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), fpath + '/actor.pth')
        torch.save(self.critic.state_dict(), fpath + '/critic.pth')

    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/actor.pth", map_location=self.device)
        critic_ckpt = torch.load(f"{fpath}/critic.pth", map_location=self.device)
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)
