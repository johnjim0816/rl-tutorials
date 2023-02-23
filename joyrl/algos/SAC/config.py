class AlgoConfig:
    def __init__(self) -> None:
        self.policy_type = 'Gaussian' # policy type
        self.lr = 1e-3 # learning rate # 3e-4
        self.gamma = 0.99 # discount factor
        self.tau = 0.005 # soft update factor
        self.alpha = 0.1 # Temperature parameter α determines the relative importance of the entropy term against the reward # 0.1
        self.automatic_entropy_tuning = False # automatically adjust α
        self.batch_size = 64 # batch size # 256
        self.hidden_dim = 64 # hidden dimension # 256
        self.n_epochs = 1 # number of epochs
        self.start_steps = 10000 # number of random steps for exploration
        self.target_update_fre = 1 # interval for updating the target network
        self.buffer_size = 1000000 # replay buffer size