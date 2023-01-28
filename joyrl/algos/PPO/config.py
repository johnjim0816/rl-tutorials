import torch
import os
print(os.path.dirname(__file__))

class AlgoConfig:
    def __init__(self):
        self.ppo_type = 'clip' # clip or kl
        if self.ppo_type == 'kl':
            self.kl_target = 0.1 # target KL divergence
            self.kl_lambda = 0.5 # lambda for KL penalty, 0.5 is the default value in the paper
            self.kl_beta = 1.5 # beta for KL penalty, 1.5 is the default value in the paper
            self.kl_alpha = 2 # alpha for KL penalty, 2 is the default value in the paper

        self.buffer_size = 20480
        self.entropy_coef = 0.01 # entropy coefficient
        self.actor_hidden_dim = 256 # hidden dimension for actor
        self.critic_hidden_dim = 256 # hidden dimension for critic
