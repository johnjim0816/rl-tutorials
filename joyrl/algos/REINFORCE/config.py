class AlgoConfig:
    def __init__(self):
        self.lr = 0.01
        self.gamma = 0.99 # discount factor
        self.hidden_dim = 36 # hidden dimension of actor 
        self.update_freq = 200 # update policy every n steps