class AlgoConfig:
    def __init__(self) -> None:
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 500 # epsilon decay rate
        self.lr = 1e-3 # learning rate 
        self.gamma = 0.99 # discount factor
        self.tau = 0.005 # soft update factor
        self.alpha = 0.1 # Temperature parameter α determines the relative importance of the entropy term against the reward # 0.1
        self.automatic_entropy_tuning = False # automatically adjust α
        self.batch_size = 64 # batch size # 256
        self.hidden_dim = 256 # hidden dimension # 256
        self.n_epochs = 1 # number of epochs
        self.target_update = 1 # interval for updating the target network
        self.buffer_size = 1000000 # replay buffer size