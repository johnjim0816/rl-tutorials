import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NoisyDQN:
    def __init__(self,model,memory,cfg) -> None:
        self.gamma = cfg.gamma  # discount factor
        self.sample_count = 0  # sample count
        self.beta_start = cfg.beta_start
        self.beta_frames = cfg.beta_frames
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.device = torch.device(cfg.device) 
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        ## copy parameters from policy net to target net
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = memory
        self.update_flag = False
    def sample_action(self, state):
        ''' sample action with e-greedy policy
        '''
        self.sample_count += 1
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_value = self.policy_net(state)
            action  = q_value.max(1)[1].item()
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
        beta = min(1.0, self.beta_start + self.sample_count * (1.0 - self.beta_start) / self.beta_frames)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights_batch, indices = self.memory.sample(self.batch_size, beta) 
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) 
        action_batch = torch.tensor(action_batch, device=self.device) 
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float)
        weights_batch = torch.tensor(weights_batch, device=self.device, dtype=torch.float)
        q_values      = self.policy_net(state_batch)
        next_q_values = self.target_net(next_state_batch)

        q_value          = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)
        
        loss  = (q_value - expected_q_value.detach()).pow(2) * weights_batch
        prios = loss + 1e-5
        loss  = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())   
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)