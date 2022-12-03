import torch
from torch.distributions import Categorical
import numpy as np
from common.models import ActorSoftmax
from common.memories import PGReplay

class Agent:
    def __init__(self,cfg) -> None:
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device) 
        self.memory = PGReplay()
        self.actor = ActorSoftmax(cfg.n_states,cfg.n_actions, hidden_dim = cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=cfg.lr)
        self.sample_count = 0
        self.update_freq = cfg.update_freq # update policy every n steps
    def sample_action(self,state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
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
        if self.sample_count % self.update_freq != 0:
            return
        print("update policy")
        state_pool,action_pool,reward_pool= self.memory.sample()
        state_pool,action_pool,reward_pool = list(state_pool),list(action_pool),list(reward_pool)
        # compute discounted rewards (Returns)
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add
        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        state = torch.tensor(state_pool, device=self.device, dtype=torch.float32)
        action = torch.tensor(action_pool, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward_pool, device=self.device, dtype=torch.float32)
        probs = self.actor(state)
        dist = Categorical(probs)
        log_probs = dist.log_prob(action)
        loss = -log_probs * reward
        loss = loss.mean()
        # Gradient Desent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()
    def save_model(self,fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), fpath+'checkpoint.pt')
    def load_model(self,fpath):
        ckpt = torch.load(f"{fpath}/checkpoint.pt", map_location=self.device)
        self.actor.load_state_dict(ckpt)