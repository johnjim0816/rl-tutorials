class AlgoConfig:
    def __init__(self) -> None:
        self.explore_steps = 1000  # exploration steps before training
        self.policy_freq = 2  # policy update frequency
        self.actor_lr = 1e-3 # actor learning rate 3e-4
        self.critic_lr = 1e-3 # critic learning rate
        self.actor_hidden_dim = 256 # actor hidden layer dimension
        self.critic_hidden_dim = 256 # critic hidden layer dimension
        self.gamma = 0.99 # discount factor
        self.tau = 0.005 # target smoothing coefficient
        self.policy_noise = 0.2 # noise added to target policy during critic update
        self.expl_noise = 0.1 # std of Gaussian exploration noise
        self.noise_clip = 0.5 # range to clip target policy noise
        self.batch_size = 100 # batch size for both actor and critic
        self.buffer_size = 1000000 # replay buffer size