class AlgoConfig:
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.critic_lr = 1e-3 # learning rate for critic
        self.actor_lr = 1e-4 # learning rate for actor
        self.buffer_size = 8000 # size of replay buffer
        self.batch_size = 128 # mini-batch size
        self.tau = 0.001 # soft update
        self.critic_hidden_dim = 256 # hidden dimension of critic
        self.actor_hidden_dim = 256 # hidden dimension of actor