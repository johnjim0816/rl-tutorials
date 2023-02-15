from config.config import DefaultConfig


class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.1 # epsilon start value
        self.epsilon_end = 0.001 # epsilon end value
        self.epsilon_decay = 0.995 # epsilon decay rate linear annealing
        self.hidden_dim = 64 # hidden_dim for MLP
        self.gamma = 0.99 # discount factor
        self.lr = 0.0001 # learning rate
        self.buffer_size = 100000 # size of replay buffer
        self.batch_size = 8 # batch size
        self.target_update = 4 # target network update frequency

        self.lookup_step = 10 # the lookup_step of DRQN
        self.min_epi_num = 16 # state moment to train the DRQN
        self.max_epi_len = 128 # max episode length
        self.max_epi_num = 100 # the max num of the buffer


        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 64],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [64, 64],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [64, 'n_actions'],
             'activation': 'none'}]
