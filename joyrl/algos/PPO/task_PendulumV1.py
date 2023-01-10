# python3
# Create Date: 2023-01-10
# Func: PPO Pendulum-v1 train
# =========================================================================
import os
from os.path import dirname
import sys
proj_path = dirname(dirname(dirname(__file__)))
if proj_path not in sys.path:
    sys.path.append(proj_path)
from config import AlgoConfig
from ppo2 import PPO
from trainer import Trainer
from common.models import ActorNormal, Critic
from common.memories import ReplayBufferQue
import torch
from tqdm.auto import tqdm
import numpy as np
import gym
print(gym.__version__) # 0.26.2


def train():
    env = gym.make('Pendulum-v1')
    cfg = AlgoConfig()
    cfg.model_dir = r'D:\TMP'
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    models = {
        'Actor': ActorNormal(n_states, n_actions, hidden_dim=cfg.actor_hidden_dim).to(torch.device(cfg.device )),
        'Critic': Critic(n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(torch.device(cfg.device ))
    }
    memory = ReplayBufferQue(cfg.buffer_size)
    ppo_agent = PPO(models, memory, cfg)
    train_helper = Trainer()
    rewards_list = []
    tq_bar = tqdm(range(400 * cfg.actor_nums))
    tq_bar.set_description('PPO(Pendulum)')
    best_rewards = -np.inf
    for i in tq_bar:
        ppo_agent, ep_reward, ep_step = train_helper.train_one_episode(env, ppo_agent, cfg)
        rewards_list.append(ep_reward)
        lst_rewards = np.mean(rewards_list[-10:])
        if (best_rewards < lst_rewards) and (i >= 11):
            best_rewards = lst_rewards
            ppo_agent.save_model(fpath=cfg.model_dir)
        
        tq_bar.set_postfix({'lastRewards': f'{lst_rewards:.3f}', 'bestRewards': f'{best_rewards:.3f}'})
    
    env.close()


def play():
    env = gym.make('Pendulum-v1')
    cfg = AlgoConfig()
    cfg.model_dir = r'D:\TMP'
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    models = {
        'Actor': ActorNormal(n_states, n_actions, hidden_dim=cfg.actor_hidden_dim).to(torch.device(cfg.device )),
        'Critic': Critic(n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(torch.device(cfg.device ))
    }
    memory = ReplayBufferQue(cfg.buffer_size)
    ppo_agent = PPO(models, memory, cfg)
    train_helper = Trainer()
    # play
    cfg.render = True
    env = gym.make('Pendulum-v1', render_mode='human')
    ppo_agent.load_model(cfg.model_dir)
    train_helper.test_one_episode(env, ppo_agent, cfg)



if __name__ == '__main__':
    # train()
    play()







