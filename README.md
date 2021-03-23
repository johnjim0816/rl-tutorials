

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



Note that ```model.py```,```memory.py```,```plot.py``` shall be utilized in different algorithms，thus they are put into ```common``` folder。

## Runnig Environment

python 3.7.9、pytorch 1.6.0、gym 0.18.0
## Usage

Environment infomations see [环境说明](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/env_info.md)

## Schedule

|                             Name                             |                      Related materials                      | Used Envs                                                    |                            Notes                             |
| :----------------------------------------------------------: | :---------------------------------------------------------: | ------------------------------------------------------------ | :----------------------------------------------------------: |
| [On-Policy First-Visit MC](https://github.com/JohnJim0816/rl-tutorials/tree/master/MonteCarlo) |                                                             | [Racetrack](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/racetrack_env.md) |                                                              |
| [Q-Learning](http://wanggithub.com/JohnJim0816/rl-tutorials/tree/master/QLearning) |                                                             | [CliffWalking-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) |                                                              |
| [Sarsa](https://github.com/JohnJim0816/rl-tutorials/tree/master/Sarsa) |                                                             | [Racetrack](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/racetrack_env.md) |                                                              |
| [DQN](https://github.com/JohnJim0816/rl-tutorials/tree/master/DQN) | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [CartPole-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) | [DQN算法实战](https://blog.csdn.net/JohnJim0/article/details/109557173) |
|                           DQN-cnn                            | [DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [CartPole-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) |              与DQN相比使用了CNN而不是全链接网络              |
| [DoubleDQN](https://github.com/JohnJim0816/rl-tutorials/tree/master/DoubleDQN) |                                                             | [CartPole-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) |                       效果不好，待改进                       |
|                       Hierarchical DQN                       |    [Hierarchical DQN](https://arxiv.org/abs/1604.06057)     |                                                              |                                                              |
| [PolicyGradient](https://github.com/JohnJim0816/rl-tutorials/tree/master/PolicyGradient) |                                                             | [CartPole-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) |                                                              |
|                             A2C                              |                                                             | [CartPole-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) |                                                              |
|                             A3C                              |                                                             |                                                              |                                                              |
|                             SAC                              |                                                             |                                                              |                                                              |
| [PPO](https://github.com/JohnJim0816/rl-tutorials/tree/master/PPO) |        [PPO paper](https://arxiv.org/abs/1707.06347)        | [CartPole-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) | [PPO算法实战](https://blog.csdn.net/JohnJim0/article/details/115126363) |
|                             DDPG                             |       [DDPG Paper](https://arxiv.org/abs/1509.02971)        | [Pendulum-v0](https://github.com/JohnJim0816/rl-tutorials/blob/master/envs/gym_info.md) |                                                              |
|                             TD3                              | [Twin Dueling DDPG Paper](https://arxiv.org/abs/1802.09477) |                                                              |                                                              |
|                             GAIL                             |                                                             |                                                              |                                                              |


## Refs


[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

[RL-Adventure](https://github.com/higgsfield/RL-Adventure)

https://www.cnblogs.com/lucifer1997/p/13458563.html
