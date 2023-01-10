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
from common.memories import PGReplay
import torch
from tqdm.auto import tqdm
import numpy as np
import gym
print(gym.__version__) # 0.26.2

def prepare_config(env):
    cfg = AlgoConfig()
    # env setting
    cfg.n_states = env.observation_space.shape[0]
    cfg.n_actions = env.action_space.shape[0]
    cfg.seed = 2023
    cfg.new_step_api = True
    cfg.continuous = True
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.max_steps = 260

    # PPO kwargs
    cfg.k_epochs = 10
    cfg.policy_clip = 0.2
    cfg.sgd_batch_size = 512
    cfg.gae_lambda = 0.9
    cfg.gamma = 0.9
    cfg.actor_nums = 3
    cfg.actor_lr = 1e-4 # learning rate for actor
    cfg.critic_lr = 5e-3 # learning rate for critic
    cfg.model_dir = r'D:\TMP'
    return cfg


def train():
    env = gym.make('Pendulum-v1')
    cfg = prepare_config(env)
    models = {
        'Actor': ActorNormal(cfg.n_states, cfg.n_actions, hidden_dim=cfg.actor_hidden_dim).to(torch.device(cfg.device )),
        'Critic': Critic(cfg.n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(torch.device(cfg.device ))
    }
    memory = PGReplay()
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
    cfg = prepare_config(env)
    models = {
        'Actor': ActorNormal(cfg.n_states, cfg.n_actions, hidden_dim=cfg.actor_hidden_dim).to(torch.device(cfg.device )),
        'Critic': Critic(cfg.n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(torch.device(cfg.device ))
    }
    memory = PGReplay()
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







