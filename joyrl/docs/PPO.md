
## 算法参数说明 

PPO算法参数如下：

```python
class AlgoConfig:
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.k_epochs = 4 # update policy for K epochs
        self.actor_lr = 0.0003 # learning rate for actor
        self.critic_lr = 0.0003 # learning rate for critic
        self.eps_clip = 0.2 # clip parameter for PPO
        self.entropy_coef = 0.01 # entropy coefficient
        self.update_freq = 100 # update policy every n steps
        self.actor_hidden_dim = 256 # hidden dimension for actor
        self.critic_hidden_dim = 256 # hidden dimension for critic
```

*  `eps_clip`：clip参数，一般设置为0.1-0.2之间即可
*  `entropy_coef`：策略熵损失系数，增加该系数提高actor的稳定性，保持0.0001-0.02即可，或者直接设置为0在一些问题中影响也不大
*  `update_freq`：更新频率，在JoyRL中设置为每隔几步更新，一般跟环境中每回合最大步数线性相关，例如carpole-v1环境中每回合最大步数是200，这里更新频率可以设置为50，100，200等等，这项参数需要根据实际经验调整
*  `k_epochs`：调整每次更新的epoch数，不能太大也不能太小，太大了一方面收敛速度会变慢，另一方面容易过拟合，太小了容易欠拟合