class AlgoConfig:
    def __init__(self) -> None:
        self.n_workers = 2 # number of workers for parallel training
        self.gamma = 0.99 # discount factor
        self.actor_lr = 3e-4 # learning rate of actor
        self.critic_lr = 1e-3 # learning rate of critic
        self.actor_hidden_dim = 256 # hidden_dim for actor MLP
        self.critic_hidden_dim = 256 # hidden_dim for critic MLP
        self.entropy_coef = 0.05 # entropy coefficient
        self.update_freq = 20 # update policy every n steps

        self.device =  "cpu"
        self.seed = 1
        self.train_eps = 1000
        self.env_name = 'CartPole-v1'
        self.max_steps = 200