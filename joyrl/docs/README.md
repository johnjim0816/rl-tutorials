# JoyRL离线版说明文档

## 目录树

JoyRL目录树如下：
```python
|-- joyrl
    |-- algos
        |-- [Algorithm name] # 指代算法名称比如DQN等
            |-- config.py # 存放每个算法的默认参数设置
                |-- class AlgoConfig # 算法参数设置的类
            |-- agent.py # 存放算法
                |-- class Agent # 每个算法的类命名为Agent
            |-- trainer.py
    |-- config 
        |-- config.py # 存放通用参数设置
    |-- presets # 预设的参数，对应的结果存放在benchmarks下面
    |-- common # 通用文件夹
        |--  memories.py # 存放经验回放相关函数或者类
        |--  models.py # 存放网络模型相关类
        |--  utils.py # 存放其他函数，比如画图、设置随机种子等等等等
    |-- envs
    |-- benchmarks # 存放训练好的结果
    |-- docs # 说明文档目录
    |-- tasks # 训练的时候会自动生成
    |-- main.py # JoyRL训练主函数
    |-- README.md # 项目README
    |-- README_cn.md # 项目中文README
    |-- requirements.txt # Pyhton依赖列表
```
## 参数说明

### 通用参数说明

通用参数目前用Python类保存，设置如下：

```python
class GeneralConfig(DefaultConfig):
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "PER_DQN" # name of algorithm
        self.new_step_api = True # whether to use new step api of gym
        self.wrapper = None # wrapper of environment
        self.render = False # whether to render environment
        self.mode = "train" # train or test
        self.seed = 0 # random seed
        self.device = "cpu" # device to use
        self.train_eps = 100 # number of episodes for training
        self.test_eps = 20 # number of episodes for testing
        self.eval_eps = 10 # number of episodes for evaluation
        self.eval_per_episode = 5 # evaluation per episode
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
```
正如项目README中使用说明所提到的，JoyRL目前使用方式有两种，一种是不带参数，直接```python main.py```，另外一种是以yaml文件的形式带参数，即```python main.py --yaml [path of yaml file]```。这里介绍一下JoyRL的运行逻辑，首先不带参数直接运行的时候，首先会加载通用参数设置，然后根据通用参数设置的算法名称加载对应算法的设置```AlgoConfig```，然后根据通用参数的self.mode即运行模式训练或者测试。由于目前JoyRL所有算法跑的默认环境都是```CartPole```或者```Pendulum```，因此如果读者在初学阶段，只需要改通用参数的环境名称或者算法名称就能体验JoyRL了。

另外就是带yaml文件参数运行的时候，JoyRL会优先设置yaml文件中的参数，比如yaml文件中环境名称是Pendulum-v1，而config.py文件中通用参数默认环境名称是CartPole-v1，这个时候yaml文件的参数会覆盖掉config.py的参数，也就是说实际运行的环境名称就是yaml文件中设置的Pendulum-v1。如果yaml文件中没有设置，那么JoyRL会自动加载默认特征。比如show_fig这种参数，即训练好之后是否在窗口展示图片，默认的设置是不展示的（因为弹出窗口总是显得有那么一点烦人XD），如果读者也是默认不展示的话那么在yaml文件中就可以不进行设置。
### 网络参数说明

略
### 算法参数说明

请跳转各自的算法说明

## 算法说明
* [Q-learning](Q-learning.md)：参考DQN说明
* [DQN](./DQN.md)
* [PPO](./PPO.md)