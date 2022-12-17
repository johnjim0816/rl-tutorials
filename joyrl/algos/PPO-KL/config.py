'''
Author: QIU clorisqiu1@gmail.com
Date: 2022-12-07 15:53:52
LastEditors: Please set LastEditors
LastEditTime: 2022-12-17 20:41:55
FilePath: /rl-tutorials/joyrl/algos/PPO-KL/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
class AlgoConfig:
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.k_epochs = 4 # update policy for K epochs
        self.actor_lr = 0.0003 # learning rate for actor
        self.critic_lr = 0.0003 # learning rate for critic

        self.kl_target = 0.1 # target KL divergence
        self.kl_lambda = 0.5 # lambda for KL penalty, 0.5 is the default value in the paper
        self.kl_beta = 1.5 # beta for KL penalty, 1.5 is the default value in the paper
        self.kl_alpha = 2 # alpha for KL penalty, 2 is the default value in the paper
        
        self.entropy_coef = 0.01 # entropy coefficient
        self.train_batch_size = 100 # update policy every n steps
        self.actor_hidden_dim = 128 # hidden dimension for actor
        self.critic_hidden_dim = 128  # hidden dimension for critic
