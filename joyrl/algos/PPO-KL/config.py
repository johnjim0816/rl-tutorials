'''
Author: QIU clorisqiu1@gmail.com
Date: 2022-12-07 15:53:52
LastEditors: QIU clorisqiu1@gmail.com
LastEditTime: 2022-12-07 15:54:37
FilePath: /rl-tutorials/joyrl/algos/PPO-KL/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
class AlgoConfig:
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.k_epochs = 4 # update policy for K epochs
        self.actor_lr = 0.0003 # learning rate for actor
        self.critic_lr = 0.0003 # learning rate for critic

        self.KL_target = 0.01 #  parameter for PPO-KL target
        self.KL_lambda = 0.5 # parameter for PPO-KL
        self.KL_beta_high = 1.5
        self.KL_beta_low = 1/self.KL_beta_high
        self.KL_alpha = 2
        
        self.entropy_coef = 0.01 # entropy coefficient
        self.update_freq = 100 # update policy every n steps
        self.actor_hidden_dim = 256 # hidden dimension for actor
        self.critic_hidden_dim = 256 # hidden dimension for critic
