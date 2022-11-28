## 0. 写在前面

本项目用于学习RL基础算法，主要面向对象为RL初学者、需要结合RL的非专业学习者，尽量做到: **注释详细**，**结构清晰**。

注意本项目为实战内容，建议首先掌握相关算法的一些理论基础，再来享用本项目，理论教程参考本人参与编写的[蘑菇书](https://github.com/datawhalechina/easy-rl)。

未来开发计划包括但不限于：多智能体算法、强化学习Python包以及强化学习图形化编程平台等等。

## 1. 项目说明

本项目内容主要包含以下几个子项目，每个子项目下都有对应的README描述：
* [Jupyter Notebook](./notebooks/)：使用Notebook写的算法，有比较详细的实战引导，推荐新手食用
* [JoyRL离线版](./joyrl/)：JoyRL离线版项目
* [PARL Tutorials](./parl_tutorials)：PARL实现强化学习代码
* [附件](./assets/)：目前包含强化学习各算法的中文伪代码

## 2. 算法环境

算法环境说明请跳转[env](./codes/envs/README.md)

## 3. 算法列表

注：点击对应的名称会跳到[codes](./codes/)下对应的算法中，其他版本还请读者自行翻阅

|                算法名称                 |                           参考文献                           |                         作者                         | 备注 |
| :-------------------------------------: | :----------------------------------------------------------: | :--------------------------------------------------: | :--: |
| [Policy Gradient](codes/PolicyGradient) | [Policy Gradient paper](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) |    [johnjim0816](https://github.com/johnjim0816)     |      |
|     [Monte Carlo](codes/MonteCarlo)     |                                                              |    [johnjim0816](https://github.com/johnjim0816)     |      |
|            [DQN](codes/DQN)             | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  |    [johnjim0816](https://github.com/johnjim0816)     |      |
|                 DQN-CNN                 |                                                              |                                                      | 待更 |
|      [DoubleDQN](codes/DoubleDQN)       |     [Double DQN Paper](https://arxiv.org/abs/1509.06461)     |    [johnjim0816](https://github.com/johnjim0816)     |      |
|     [DuelingDQN](codes/DuelingDQN)      |     [DuelingDQN Paper](https://arxiv.org/abs/1511.06581)     |    [johnjim0816](https://github.com/johnjim0816)     |      |
|        [PER_DQN](codes/PER_DQN)         |      [PER DQN Paper](https://arxiv.org/abs/1511.05952)       | [wangzhongren](https://github.com/wangzhongren-code) |      |
|       [NoisyDQN](codes/NoisyDQN)        |     [Noisy DQN Paper](https://arxiv.org/abs/1706.10295)      |    [johnjim0816](https://github.com/johnjim0816)     |      |
|          [SoftQ](codes/SoftQ)           |  [Soft Q-learning paper](https://arxiv.org/abs/1702.08165)   |    [johnjim0816](https://github.com/johnjim0816)     |      |
|            [SAC](codes/SAC)             |      [SAC paper](https://arxiv.org/pdf/1812.05905.pdf)       |                                                      |      |
|        [SAC-Discrete](codes/SAC)        |  [SAC-Discrete paper](https://arxiv.org/pdf/1910.07207.pdf)  |                                                      |      |
|                  SAC-S                  |       [SAC-S paper](https://arxiv.org/abs/1801.01290)        |                                                      |      |
|                  DSAC                   | [DSAC paper](https://paperswithcode.com/paper/addressing-value-estimation-errors-in) |                                                      | 待更 |

## 4. 友情说明

推荐使用VS Code做项目，入门可参考[VSCode上手指南](https://blog.csdn.net/JohnJim0/article/details/126366454)
