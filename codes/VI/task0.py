#!/usr/bin/env python
# coding=utf-8
'''
Author: GuoShiCheng
Email: guoshichenng@gmail.com
Date: 2022-11-19 09:56:33
LastEditor: GuoShiCheng
LastEditTime: 2022-11-19 09:56:33
Discription: theAlley,walkInThePark
'''

'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-12 11:09:54
LastEditor: JiangJi
LastEditTime: 2022-10-31 00:13:31
Discription: CartPole-v1,Acrobot-v1
'''
import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path
import gym
from common.utils import all_seed,merge_class_attrs
from common.models import MLP
from common.memories import ReplayBuffer
from common.launcher import Launcher
from envs.register import register_env
from envs.simple_grid import DrunkenWalkEnv
from value_iteration import VI
from config.config import GeneralConfigVI,AlgoConfigVI
class Main(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'], GeneralConfigVI())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'], AlgoConfigVI())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''
        env = DrunkenWalkEnv(map_name=cfg.env_name)
        if cfg.seed !=0: # set random seed
            all_seed(env,seed=cfg.seed) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
            env_P = env.P # state transfer probability
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        logger.info(f"n_states: {n_states}, n_actions: {n_actions}") # print info
        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        setattr(cfg, 'env_P', env_P)
        agent = VI(cfg)  # create agent
        return env, agent
    def train_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
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
            next_state, reward, terminated, info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step


if __name__ == "__main__":
    main = Main()
    main.run()
    

