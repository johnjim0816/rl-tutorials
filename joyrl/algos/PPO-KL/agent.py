import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from common.models import ActorSoftmax, Critic
from common.memories import PGReplay

class Agent:
    def __init__(self,cfg) -> None:
        
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device) 
        self.actor = ActorSoftmax(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states,1,hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.action_space = cfg.action_space

        self.kl_target = cfg.kl_target #  parameter for PPO-KL target
        self.kl_lambda = cfg.kl_lambda # parameter for PPO-KL
        self.kl_beta = cfg.kl_beta
        self.kl_alpha = cfg.kl_alpha

        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sample_count = 0
        self.train_batch_size = cfg.train_batch_size

    def sample_action(self,state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.probs = probs.detach()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()
    def update(self):
        # update policy every n steps
        if self.sample_count % self.train_batch_size != 0:
            return
        # print("update policy")
        old_states, old_actions, old_rewards, old_dones,old_probs,old_log_probs = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32) # shape: [train_batch_size,n_states]
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_probs = torch.cat(old_probs).to(self.device)
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
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for _ in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            new_probs = self.actor(old_states)
            dist = Categorical(new_probs)
            # get new action probabilities
            new_log_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_log_probs - old_log_probs).unsqueeze(dim=1) # old_log_probs must be detached, shape: [train_batch_size, 1]
            # compute surrogate loss
            surr1 = ratio * advantage # [train_batch_size,1]
            kl_mean = F.kl_div(torch.log(new_probs.detach()), old_probs.unsqueeze(1),reduction='mean') # KL(input|target),new_probs.shape: [train_batch_size, n_actions]
            # kl_div = torch.mean(new_probs * (torch.log(new_probs) - torch.log(old_probs)), dim=1) # KL(new|old),new_probs.shape: [train_batch_size, n_actions]
            surr2 = self.kl_lambda * kl_mean
            # surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = - (surr1.mean() + surr2 + self.entropy_coef * dist.entropy().mean())
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            if kl_mean > self.kl_beta * self.kl_target:
                self.kl_lambda *= self.kl_alpha
            elif kl_mean < 1/self.kl_beta * self.kl_target:
                self.kl_lambda /= self.kl_alpha

        self.memory.clear()
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{fpath}/actor.pth")
        torch.save(self.critic.state_dict(), f"{fpath}/critic.pth")
    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/actor.pth", map_location=self.device)
        critic_ckpt = torch.load(f"{fpath}/critic.pth", map_location=self.device)
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)