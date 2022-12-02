
## 算法参数说明

DQN的算法参数如下：

```python
class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate
        self.gamma = 0.95  # discount factor
        self.lr = 0.0001  # learning rate
        self.buffer_size = 100000  # size of replay buffer
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 'n_actions'],
             'activation': 'none'}]
```
其中value_layers设置部分请参考网络参数说明，这里略过。gamma是强化学习中的折扣因子，一般调整在0.9-0.999之间即可，可以默认为0.99。除了网络参数设置之外，DQN参数调整的空间较少。比如batch_size跟深度学习一样，一般都在64，128和256之间(太大了训练的物理机器吃不消)。buffer_size、target_update以及epsilon都需要根据实际环境的情况来经验性的调整。

这里着重一下epsilon的衰减机制，也就是探索率相关，在JoyRL中目前是以指数方式衰减的，如下：
```python
self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
```
转成数学公式如下：

$$
\varepsilon = (\varepsilon_{start}-\varepsilon_{end}) * e ^{- \frac{sample\_count}{epsilon\_decay}} + \varepsilon_{end}
$$

训练开始的时候$sample\_count$等于0，则$\varepsilon = \varepsilon_{start}$，相当于$\varepsilon_{start}$的概率进行随机策略，随着$sample\_count$逐渐增大，指数项$e ^{- \frac{sample\_count}{epsilon\_decay}}$就会逐渐趋近于0，最后就会接近于$\varepsilon_{end}$，也就是较小的探索率。因此这里的$epsilon\_decay$是比较重要的，跟环境的每回合最大步数和读者设置的训练回合数有关，或者说跟训练预估的$总步数=环境的每回合最大步数*读者设置的训练回合数$有关，因此需要有一个合理的设置，不要让指数项太快地趋近于0，此时会导致没有进行足够的随机探索。也不要让指数项等到训练结束了或者说到达总步数了还没有趋近于0，此时会导致整个训练过程中随机探索的部分占比过大，影响算法的收敛。