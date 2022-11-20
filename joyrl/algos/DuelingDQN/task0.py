#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-12 11:09:54
LastEditor: JiangJi
LastEditTime: 2022-11-15 00:11:24
Discription: CartPole-v1,Acrobot-v1
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path
import gym
import torch.nn as nn
from common.utils import all_seed,merge_class_attrs
from common.models import MLP
from common.memories import ReplayBuffer
from common.launcher import Launcher
from envs.register import register_env
from dueling_dqn import DuelingDQN
from config.config import GeneralConfigDuelingDQN,AlgoConfigDuelingDQN

class DuelingNet(nn.Module):
    def __init__(self, n_states, n_actions,hidden_dim=128):
        super(DuelingNet, self).__init__()
        
        # hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU()
        )
        
        #  advantage
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # value
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        x = self.hidden_layer(state)
        advantage = self.advantage_layer(x)
        value     = self.value_layer(x)
        return value + advantage - advantage.mean()

class Main(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],GeneralConfigDuelingDQN())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],AlgoConfigDuelingDQN())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name,new_step_api=True)  # create env
        all_seed(env,seed=cfg.seed) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        logger.info(f"n_states: {n_states}, n_actions: {n_actions}") # print info
        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        # cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        model = DuelingNet(n_states,n_actions,hidden_dim=cfg.hidden_dim)
        memory =  ReplayBuffer(cfg.buffer_size) # replay buffer
        agent = DuelingDQN(model,memory,cfg)  # create agent
        return env, agent
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            agent.memory.push(state, action, reward,
                            next_state, terminated)  # save transitions
            agent.update()  # update agent
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step

if __name__ == "__main__":
    main = Main()
    main.run()

