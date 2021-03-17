#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2021-03-17 20:20:26
@Discription: 
@Environment: python 3.7.9
'''
import sys,os
sys.path.append(os.getcwd()) # add current terminal path
import torch
import gym
import datetime

from A2C.agent import A2C
from A2C.env import make_envs
from common.plot import plot_rewards
from common.utils import save_results



SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/' # 生成保存的模型路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"): 
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH): 
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # 存储reward的路径
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"): 
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH): 
    os.mkdir(RESULT_PATH)

class A2CConfig:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 3e-4 # learnning rate
        self.actor_lr = 1e-4 # learnning rate of actor network
        self.memory_capacity = 10000 # capacity of replay memory
        self.batch_size = 128
        self.train_eps = 4000
        self.train_steps = 5
        self.eval_eps = 200
        self.eval_steps = 200
        self.target_update = 4
    

def test_env(agent,device='cpu'):
    env = gym.make("CartPole-v0")
    state = env.reset()
    ep_reward=0
    for _ in range(200):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, value = agent.model(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state
        ep_reward += reward
        if done:
            break
    return ep_reward


def train(cfg):
    print('Start to train ! \n')
    envs = make_envs(num_envs=16,env_name="CartPole-v0")
    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = A2C(state_dim, action_dim, hidden_dim=256)
    # moving_average_rewards = []
    # ep_steps = []
    state = envs.reset()
    for i_episode in range(cfg.train_eps):
        log_probs = []
        values    = []
        rewards = []
        masks     = []
        entropy = 0
        for i_step in range(cfg.train_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = agent.model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            state = next_state                                                  
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        if i_episode%20 == 0:
            print("reward",test_env(agent,device='cpu'))
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value =agent.model(next_state)
        returns = agent.compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
 
        
        # print('Episode:', i_episode, ' Reward: %i' %
        #       int(ep_reward[0]), 'n_steps:', i_step)
        # ep_steps.append(i_step)
        # rewards.append(ep_reward)
        # if i_episode == 1:
        #     moving_average_rewards.append(ep_reward[0])
        # else:
        #     moving_average_rewards.append(
        #         0.9*moving_average_rewards[-1]+0.1*ep_reward[0])
        # writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        # writer.add_scalar('steps_of_each_episode',
        #                   ep_steps[-1], i_episode)
    print('Complete training！')
    ''' 保存模型 '''
    # save_model(agent,model_path=SAVED_MODEL_PATH)
    # '''存储reward等相关结果'''
    # save_results(rewards,moving_average_rewards,ep_steps,tag='train',result_path=RESULT_PATH)


if __name__ == "__main__":
    cfg = A2CConfig()
    train(cfg)

