#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-12 11:09:54
LastEditor: JiangJi
LastEditTime: 2022-11-16 06:45:38
Discription: CartPole-v1,Acrobot-v1
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.utils import all_seed,merge_class_attrs
from common.memories import PrioritizedReplayBuffer
from common.launcher import Launcher
from envs.register import register_env
from noisy_dqn import NoisyDQN
from config.config import GeneralConfigNoisyDQN,AlgoConfigNoisyDQN
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(torch.tensor(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(torch.tensor(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
class NoisyNet(nn.Module):
    def __init__(self, n_states, n_actions,hidden_dim=128):
        super(NoisyNet, self).__init__()
        
        self.linear =  nn.Linear(n_states, hidden_dim)
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy2 = NoisyLinear(hidden_dim, n_actions)
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
class Main(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],GeneralConfigNoisyDQN())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],AlgoConfigNoisyDQN())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name,new_step_api=True)  # create env
        all_seed(env,seed=cfg.seed) # set random seed
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
        model = NoisyNet(n_states,n_actions,hidden_dim=cfg.hidden_dim)
        memory =  PrioritizedReplayBuffer(cfg.buffer_size,alpha=cfg.alpha) # replay buffer
        agent = NoisyDQN(model,memory,cfg)  # create agent
        return env, agent
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            agent.memory.push(state, action, reward, next_state, terminated)  # save transitions
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
    # def train(self,env, agent,cfg,logger):
    #     ''' 训练
    #     '''
    #     logger.info("Start training!")
    #     logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
    #     rewards = []  # record rewards for all episodes
    #     steps = [] # record steps for all episodes
    #     for i_ep in range(cfg.train_eps):   
    #         ep_reward = 0  # reward per episode
    #         ep_step = 0
    #         state = env.reset()  # reset and obtain initial state
    #         for _ in range(cfg.max_steps):
    #             ep_step += 1
    #             action = agent.sample_action(state)  # sample action
    #             next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
    #             agent.memory.push(state, action, reward,
    #                             next_state, terminated)  # save transitions
    #             state = next_state  # update next state for env
    #             agent.update()  # update agent
    #             ep_reward += reward  #
    #             if terminated:
    #                 break
    #         if (i_ep + 1) % cfg.target_update == 0:  # target net update, target_update means "C" in pseucodes
    #             agent.target_net.load_state_dict(agent.policy_net.state_dict())
    #         steps.append(ep_step)
    #         rewards.append(ep_reward)
    #         logger.info(f'Episode: {i_ep+1}/{cfg.train_eps}, Reward: {ep_reward:.2f}: Epislon: {agent.epsilon:.3f}')
    #     logger.info("Finish training!")
    #     env.close()
    #     res_dic = {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
    #     return res_dic

    # def test(self,cfg, env, agent,logger):
    #     logger.info("Start testing!")
    #     logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
    #     rewards = []  # record rewards for all episodes
    #     steps = [] # record steps for all episodes
    #     for i_ep in range(cfg.test_eps):
    #         ep_reward = 0  # reward per episode
    #         ep_step = 0
    #         state = env.reset()  # reset and obtain initial state
    #         for _ in range(cfg.max_steps):
    #             ep_step+=1
    #             action = agent.predict_action(state)  # predict action
    #             next_state, reward, terminated, _, _ = env.step(action)  
    #             state = next_state  
    #             ep_reward += reward 
    #             if terminated:
    #                 break
    #         steps.append(ep_step)
    #         rewards.append(ep_reward)
    #         logger.info(f"Episode: {i_ep+1}/{cfg.test_eps}, Reward: {ep_reward:.2f}")
    #     logger.info("Finish testing!")
    #     env.close()
    #     return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}


if __name__ == "__main__":
    main = Main()
    main.run()

