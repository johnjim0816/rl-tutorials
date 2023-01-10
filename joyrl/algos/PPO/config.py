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
        self.model_dir = os.getcwd()
        # env setting
        self.continuous = True # continuous action space
        self.seed = 2023
        self.new_step_api = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_steps = 260
        
        # PPO kwargs
        self.k_epochs = 10  # update policy for K epochs
        self.policy_clip = 0.2
        self.train_batch_size = 2048 # ppo train batch size
        self.sgd_batch_size = 512 # sgd batch size
        self.gae_lambda = 0.9
        self.gamma = 0.9 # discount factor
        self.actor_nums = 3
        self.actor_lr = 1e-4 # learning rate for actor
        self.critic_lr = 5e-3 # learning rate for critic
        self.eps_clip = 0.2 # clip parameter for PPO
        