import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import math
from common.models import ActorSoftmax, Critic
from common.memories import PGReplay

# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         # State initialization
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)

#                 # share in memory
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()

class SharedAdam(torch.optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad,alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad,value = 1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom,value = -step_size)
        return loss
class Agent:
    def __init__(self, cfg, is_share_agent = False):
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef
        self.device = torch.device(cfg.device) 
        self.actor = ActorSoftmax(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states,1,hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        if is_share_agent: # the agent has no share agent, which means this agent itself is share agent
            self.agent_name = 'share' # share or local
            self.actor.share_memory()
            self.critic.share_memory()
            # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
            # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
            self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=cfg.actor_lr)
            self.critic_optimizer = SharedAdam(self.critic.parameters(), lr=cfg.critic_lr)
            self.actor_optimizer.share_memory()
            self.critic_optimizer.share_memory()
        self.memory = PGReplay()
        self.sample_count = 0
        self.update_freq = cfg.update_freq
    def sample_action(self, state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()
    def update(self,next_state,terminated,share_agent=None):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")
        states, actions, rewards, dones = self.memory.sample()
        # convert to tensor
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32)
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32)
        # compute returns
        # compute returns
        if not terminated:
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            next_value = self.critic(next_state).detach()
        else:
            next_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        returns = self.compute_returns(next_value,rewards,dones)
        values = self.critic(states)
        advantages = returns - values.detach()
        probs = self.actor(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions).unsqueeze(dim=1) # log_probs.shape = (batch_size,1), which is the same as advantages.shape
        actor_loss = (-log_probs*advantages).mean()+ self.entropy_coef * dist.entropy().mean()
        critic_loss = (returns - values).pow(2).mean()
        # self.actor_optimizer.zero_grad()
        # self.critic_optimizer.zero_grad()
        # tot_loss.backward()
        # self.actor_optimizer.step()
        # self.critic_optimizer.step()
        if share_agent is not None:
            share_agent.actor_optimizer.zero_grad()
            share_agent.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            for param, share_param in zip(self.actor.parameters(), share_agent.actor.parameters()):
                # if share_param.grad is None:
                #     share_param._grad = param.grad
                share_param._grad = param.grad
            for param, share_param in zip(self.critic.parameters(), share_agent.critic.parameters()):
                # if share_param.grad is None:
                #     share_param._grad = param.grad
                share_param._grad = param.grad
            share_agent.actor_optimizer.step()
            share_agent.critic_optimizer.step()
            self.actor.load_state_dict(share_agent.actor.state_dict())
            self.critic.load_state_dict(share_agent.critic.state_dict())
        else:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        # clear memory
        self.memory.clear()
    def compute_returns(self, next_value, rewards, dones):
        '''monte carlo estimate of state rewards'''
        returns = torch.zeros_like(rewards)
        R = next_value
        for t in reversed(range(len(rewards))):

            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        # Normalizing the rewards:
        returns = returns.clone().unsqueeze(1).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by 
        return returns
        
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{fpath}/actor_checkpoint.pt")
        torch.save(self.critic.state_dict(), f"{fpath}/critic_checkpoint.pt")

    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/actor_checkpoint.pt")
        critic_ckpt = torch.load(f"{fpath}/critic_checkpoint.pt")
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)