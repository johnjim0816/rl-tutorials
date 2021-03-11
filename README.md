

## 写在前面

本项目用于学习RL基础算法，尽量做到：

* 注释详细
* 结构清晰
  
  代码结构清晰，主要分为以下几个脚本：

  * ```env.py``` 用于构建强化学习环境，也可以重新normalize环境，比如给action加noise
  * ```model.py``` 强化学习算法的基本模型，比如神经网络，actor，critic等
  * ```memory.py``` 保存Replay Buffer，用于off-policy
  * ```agent.py``` RL核心算法，比如dqn等，主要包含update和select_action两个方法，
  * ```main.py``` 运行主函数
  * ```params.py``` 保存各种参数
  * ```plot.py``` 利用matplotlib或seaborn绘制rewards图，包括滑动平均的reward，结果保存在result文件夹中

## 运行环境

python 3.7.9

pytorch 1.6.0

tensorboard 2.3.0 

torchvision 0.7.0 

gym 0.17.3
## 使用说明

本仓库使用到的环境信息请跳转[环境说明](https://github.com/JohnJim0816/reinforcement-learning-tutorials/blob/master/env_info.md), 在各算法目录下也有相应说明(比如如何运行程序等)

## Value-based



### DQN

[DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)


### DQN-cnn

[DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

跟DQN一样，只不过采取图像作为state，因此使用CNN网络而不是普通的全连接网络(MLP)

[Hierarchical DQN](https://arxiv.org/abs/1604.06057)

## Policy-based

### DDPG

[DDPG Paper](https://arxiv.org/abs/1509.02971)

能够输出连续动作

## TD3

[Twin Dueling DDPG Paper](https://arxiv.org/abs/1802.09477)

## Refs


[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

[RL-Adventure](https://github.com/higgsfield/RL-Adventure)
