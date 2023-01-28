| 整体分类 |    任务说明    |                             备注                             |
| -------- | :------------: | :----------------------------------------------------------: |
| 框架功能 |  实现CNN功能   | 目前已有DQN-CNN但不稳定，需要在其他算法以及环境(breakout、蒙特祖玛)实现 |
|          |  实现RNN功能   |                         例如DRQN算法                         |
|          |  实现多头输入  |         关键在于自注意力机制以及寻找有价值的实现环境         |
|          |   实现多线程   | 目前在A3C上用multiprocessing实现，需要迁移到其他所有算法(要更改主框架) |
|          |  TF1 backend   |                 基于tensorflow1实现相关算法                  |
| 算法实现 |    基础算法    | TRPO、蒙特卡洛算法优化(目前运行特别慢，包含了太多for循环，，，) |
|          |  多智能体算法  | QMIX算法、VDN算法、MAPPO算法，场景：MPE、gfootball([简单3v3](https://github.com/johnjim0816/gfootball)) |
|          |   离线RL算法   |                                                              |
|          |    ICM算法     |                                                              |
|          |  模仿学习算法  | BC+TD3、LfHF算法(参考repo：https://github.com/mrahtz/learning-from-human-preferences) |
| 场景应用 |  推荐系统场景  | 可在[RL4RS](https://github.com/fuxiAIlab/RL4RS)上实现强化学习加推荐系统的结合 |
|          |    股票场景    |                                                              |
|          |  自动驾驶场景  |                                                              |
|          | 医疗场景(张怡) |                                                              |
|          |                |                                                              |



