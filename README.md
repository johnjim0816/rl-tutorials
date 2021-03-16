

[Eng](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/README.md)|[中文](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/README_cn.md)

## Introduction

This repo is used to learn basic RL algorithms, we will make it **detailed comment** and **clear structure** as much as possible:

The code structure mainly contains several scripts as following：

* ```model.py``` basic network model of RL, like MLP, CNN
* ```memory.py``` Replay Buffer
* ```plot.py``` use seaborn to plot rewards curve，saved in folder ``` result```.
* ```env.py``` to custom or normalize environments
* ```agent.py``` core algorithms, include a python Class with functions(choose action, update)
* ```main.py``` main function



Note that ```model.py```,```memory.py```,```plot.py``` shall be utilized in different algorithms，thus they are put into ```common folder。

## Runnig Environment

python 3.7.9、pytorch 1.6.0、gym 0.18.0
## Usage

Environment infomations see [环境说明](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/env_info.md)

## Schedule

|           Name           |                      Related materials                      |             Note              | Schedule |
| :----------------------: | :---------------------------------------------------------: | :---------------------------: | :------: |
| On-Policy First-Visit MC |                                                             |                               |    OK    |
|        Q-Learning        |                                                             |                               |    OK    |
|          SARSA           |                                                             |                               |    OK    |
|           DQN            | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) |                               |    OK    |
|         DQN-cnn          | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | use CNN compare to normal DQN |    OK    |
|        DoubleDQN         |                                                             |        Need to improve        |    OK    |
|     Hierarchical DQN     |    [Hierarchical DQN](https://arxiv.org/abs/1604.06057)     |                               |          |
|      PolicyGradient      |                                                             |                               |    OK    |
|           A2C            |                                                             |                               |    OK    |
|           A3C            |                                                             |                               |          |
|           PPO            |                                                             |                               |          |
|           DDPG           |       [DDPG Paper](https://arxiv.org/abs/1509.02971)        |                               |    OK    |
|           TD3            | [Twin Dueling DDPG Paper](https://arxiv.org/abs/1802.09477) |                               |          |


## Refs


[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

[RL-Adventure](https://github.com/higgsfield/RL-Adventure)

https://www.cnblogs.com/lucifer1997/p/13458563.html
