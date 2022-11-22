
from common.config import GeneralConfig,AlgoConfig
import numpy as np
import gym
import torch
import torch.nn as nn

class GeneralConfigBC(GeneralConfig):
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "BC" # name of algorithm
        self.mode = "train" # train or test
        self.seed = 1 # random seed
        self.device = "cuda" # device to use
        self.train_eps = 100 # number of episodes for training
        self.max_steps = 500 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
        self.render = True
        
class AlgoConfigBC(AlgoConfig):
    def __init__(self) -> None:
        self.image_shape = [128, 128]
        self.input_dim = 4
        self.image_obs = False
        self.lr = 5e-5 # learning rate
        self.batch_size = 512 # batch size
        self.continuous_action = False 
        self.hidden = [512, 256]
        self.model_path = './tasks/CartPole-v1_BC_20221122-013054/models'



class MergedConfig:
    def __init__(self) -> None:
        pass