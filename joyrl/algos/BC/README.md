<!--
 * @Author: = shikejianalan@gmail.com
 * @Date: 2022-11-19 22:30:51
 * @LastEditors: = shikejianalan@gmail.com
 * @LastEditTime: 2022-11-22 01:44:08
 * @FilePath: \rl-tutorials\codes\BC\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## BC Algorithm

To run the algorithm, please first cd to the correct working direction:

```
cd ./rl_tutorial/codes/BC
```
and then run the main.py file
```
python main.py
```
The hyperparameters and configs can be modified in [config](./config/config.py). 

This algorithm is designed for Cartpole-v1 environment. We first trained an PPO agent to create demonstrations
```
ppo_expert = PPO('MlpPolicy', 'CartPole-v1')
ppo_expert.learn(total_timesteps=3e4, eval_freq=10000)
```
Then we train a simple fully connected neural network to fit the data in demonstrations. The learnt agent can perform very well on the CartPole-v1 environment. 

![](materials/gym_animation.gif)
