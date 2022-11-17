#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-10-30 00:37:33
LastEditor: JiangJi
LastEditTime: 2022-10-30 01:19:08
Discription: default parameters of DQN
'''
from common.config import GeneralConfig,AlgoConfig
class GeneralConfigDQN(GeneralConfig):
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "NoisyDQN" # name of algorithm
        self.mode = "test" # train or test
        self.seed = 0 # random seed
        self.device = "cuda" # device to use
        self.train_eps = 100 # number of episodes for training
        self.test_eps = 10 # number of episodes for testing
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = True
        self.load_path = "Train_CartPole-v1_NoisyDQN_20221117-223548" # path to load model # Train_CartPole-v1_NoisyDQN_20221117-223548 tasks
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
        
class AlgoConfigDQN(AlgoConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 500 # epsilon decay rate
        self.hidden_dim = 256 # hidden_dim for MLP
        self.gamma = 0.95 # discount factor
        self.lr = 0.0001 # learning rate
        self.buffer_size = 100000 # size of replay buffer
        self.batch_size = 64 # batch size
        self.target_update = 4 # target network update frequency
