

## 写在前面

【待完善】

本项目用于学习RL基础算法，尽量做到：

* 注释详细
* 结构清晰
  
  代码结构清晰，主要分为以下几个脚本：

  * ```env.py``` 用于构建强化学习环境，也可以重新normalize环境，比如给action加noise
  * ```model.py``` 强化学习算法的基本模型，比如神经网络，actor，critic等
  * ```memory.py``` 保存Replay Buffer，用于off-policy
  * ```agent.py``` RL核心算法，比如dqn等，主要包含update和select_action两个方法，
  * ```main.py``` 运行主函数
  * ```plot.py``` 利用matplotlib或seaborn绘制rewards图，包括滑动平均的reward，结果保存在result文件夹中

## 运行环境

python 3.7.9

pytorch 1.6.0

tensorboard 2.3.0 

torchvision 0.7.0 

gym 0.17.3

## gym环境说明

### [CartPole v0](https://github.com/openai/gym/wiki/CartPole-v0)

<img src="assets/image-20200820174307301.png" alt="image-20200820174307301" style="zoom:50%;" />

通过向左或向右推车能够实现平衡，所以动作空间由两个动作组成。每进行一个step就会给一个reward，如果无法保持平衡那么done等于true，本次episode失败。理想状态下，每个episode至少能进行200个step，也就是说每个episode的reward总和至少为200，step数目至少为200

### [Pendulum-v0](https://github.com/openai/gym/wiki/Pendulum-v0)

<img src="assets/image-20200820174814084.png" alt="image-20200820174814084" style="zoom:50%;" />

钟摆以随机位置开始，目标是将其摆动，使其保持向上直立。动作空间是连续的，值的区间为[-2,2]。每个step给的reward最低为-16.27，最高为0。目前最好的成绩是100个episode的reward之和为-123.11 ± 6.86。

## Value-based



### DQN

[DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)


### DQN-cnn

[DQN-paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

跟DQN一样，只不过采取图像作为state，因此使用CNN网络而不是普通的全连接网络(FCN)

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