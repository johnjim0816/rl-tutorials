import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path


import gym
import numpy as np
import torch
import torch.optim as optim
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
from common.multiprocessing_env import SubprocVecEnv
from A2C.model import ActorCritic

class A2CConfig:
    def __init__(self) -> None:
        self.algo='A2C'
        self.env= 'CartPole-v0'
        self.n_envs = 8
        self.gamma = 0.99
n_envs = 8
env_name = "CartPole-v0"
def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk

envs = [make_env() for i in range(n_envs)]
envs = SubprocVecEnv(envs) # 8 env
env = gym.make(env_name) # a single env

def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


state_dim  = envs.observation_space.shape[0]
action_dim = envs.action_space.n

#Hyper params:
hidden_size = 256
lr          = 1e-3
n_steps   = 5

model = ActorCritic(state_dim, action_dim, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())

max_frames   = 20000
frame_idx    = 0
test_rewards = []
state = envs.reset()
while frame_idx < max_frames:
    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0
    # rollout trajectory
    
    for _ in range(n_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        state = next_state
        frame_idx += 1
        if frame_idx % 100 == 0:
            test_reward = np.mean([test_env() for _ in range(10)])
            print(f"frame_idx:{frame_idx}, test_reward:{test_reward}")
            test_rewards.append(test_reward)
            # plot(frame_idx, test_rewards)   
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)
    advantage = returns - values
    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    pass