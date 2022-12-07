class AlgoConfig:
    def __init__(self):
        self.gamma = 0.99  # discount factor
        self.k_epochs = 4  # update policy for K epochs
        self.actor_lr = 0.0003  # learning rate for actor
        self.critic_lr = 0.0003  # learning rate for critic
        self.eps_clip = 0.2  # clip parameter for PPO
        self.entropy_coef = 0.01  # entropy coefficient
        self.update_freq = 2048  # update policy every n steps
        self.actor_hidden_dim = 256  # hidden dimension for actor
        self.critic_hidden_dim = 256  # hidden dimension for critic
        self.batch_size = 2048
