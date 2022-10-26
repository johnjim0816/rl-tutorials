#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-12 11:09:54
LastEditor: JiangJi
LastEditTime: 2022-10-26 09:48:37
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
from common.config import DefaultConfig
from envs.register import register_env
from dqn import DQN
from config.config import GeneralConfigDQN,AlgoConfigDQN

class GeneralConfig(DefaultConfig):
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.seed = 0 # random seed
        self.device = "cuda" # device to use
        self.train_eps = 200 # number of episodes for training
        self.test_eps = 10 # number of episodes for testing
        self.max_steps = 200 # max steps for each episode
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
        
class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 500 # epsilon decay rate
        self.gamma = 0.95 # discount factor
        self.lr = 0.0001 # learning rate
        self.buffer_size = 100000 # size of replay buffer
        self.batch_size = 64 # batch size
        self.target_update = 4 # target network update frequency

class Main(Launcher):
    # def get_cfg(self):
    #     """ hyperparameters
    #     """
    #     cfgs = {'cfg':Config(),'general_cfg':GeneralConfig(),'algo_cfg':AlgoConfig()}  # create config object      
    #     return cfgs
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],GeneralConfigDQN())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],AlgoConfigDQN())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name,new_step_api=True)  # create env
        if cfg.seed !=0: # set random seed
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
        model = MLP(n_states,n_actions,hidden_dim=cfg.hidden_dim)
        memory =  ReplayBuffer(cfg.buffer_size) # replay buffer
        agent = DQN(model,memory,cfg)  # create agent
        return env, agent

    def train(self,cfg, env, agent,logger):
        ''' 训练
        '''
        logger.info("Start training!")
        logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = []  # record rewards for all episodes
        steps = []
        for i_ep in range(cfg.train_eps):   
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            for _ in range(cfg.max_steps):
                ep_step += 1
                action = agent.sample_action(state)  # sample action
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
                agent.memory.push(state, action, reward,
                                next_state, terminated)  # save transitions
                state = next_state  # update next state for env
                agent.update()  # update agent
                ep_reward += reward  #
                if terminated:
                    break
            if (i_ep + 1) % cfg.target_update == 0:  # target net update, target_update means "C" in pseucodes
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            steps.append(ep_step)
            rewards.append(ep_reward)
            logger.info(f'Episode: {i_ep+1}/{cfg.train_eps}, Reward: {ep_reward:.2f}: Epislon: {agent.epsilon:.3f}')
        logger.info("Finish training!")
        env.close()
        res_dic = {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
        return res_dic

    def test(self,cfg, env, agent,logger):
        logger.info("Start testing!")
        logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = []  # record rewards for all episodes
        steps = []
        for i_ep in range(cfg.test_eps):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            for _ in range(cfg.max_steps):
                ep_step+=1
                action = agent.predict_action(state)  # predict action
                next_state, reward, terminated, _, _ = env.step(action)  
                state = next_state  
                ep_reward += reward 
                if terminated:
                    break
            steps.append(ep_step)
            rewards.append(ep_reward)
            logger.info(f"Episode: {i_ep+1}/{cfg.test_eps}, Reward: {ep_reward:.2f}")
        logger.info("Finish testing!")
        env.close()
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}


if __name__ == "__main__":
    main = Main()
    main.run()

