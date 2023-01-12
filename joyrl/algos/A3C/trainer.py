#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-12-03 19:40:32
LastEditor: JiangJi
LastEditTime: 2023-01-11 13:28:58
Discription: 
'''
import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
p_parent_path = os.path.dirname(parent_path)
sys.path.append(p_parent_path)  # add path to system path
import torch.multiprocessing as mp
import gym
from common.utils import all_seed,check_n_workers,plot_rewards
from common.models import ActorSoftmax, Critic


class Worker(mp.Process):
    def __init__(self,cfg,worker_id,share_agent,env,local_agent, global_ep = None,global_r_que = None,global_best_reward = None):
        super(Worker,self).__init__()
        self.mode = cfg.mode
        self.worker_id = worker_id
        self.global_ep = global_ep
        self.global_r_que = global_r_que
        self.global_best_reward = global_best_reward

        self.share_agent = share_agent
        self.local_agent = local_agent
        self.env = env 
        self.seed = cfg.seed
        self.worker_seed = cfg.seed + worker_id
        self.train_eps = cfg.train_eps
        self.test_eps = cfg.test_eps
        self.max_steps = cfg.max_steps
        self.eval_eps = cfg.eval_eps
        self.model_dir = cfg.model_dir
    def train(self):
        while self.global_ep.value <= self.train_eps:
            state = self.env.reset(seed = self.worker_seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            while True:
                ep_step += 1
                action = self.local_agent.sample_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.local_agent.memory.push((state, action, reward, terminated))
                self.local_agent.update(next_state,terminated,share_agent=self.share_agent)
                state = next_state
                ep_r += reward
                if terminated or ep_step >= self.max_steps:
                    print(f"Worker {self.worker_id} finished episode {self.global_ep.value} with reward {ep_r:.3f}")
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    self.global_r_que.put(ep_r)
                    break
            if (self.global_ep.value+1) % self.eval_eps == 0:
                mean_eval_reward = self.evaluate()
                if mean_eval_reward > self.global_best_reward.value:
                    self.global_best_reward.value = mean_eval_reward
                    self.share_agent.save_model(self.model_dir)
                    print(f"Worker {self.worker_id} saved model with current best eval reward {mean_eval_reward:.3f}")
        self.global_r_que.put(None)
    def test(self):
        while self.global_ep.value <= self.test_eps:
            state = self.env.reset(seed = self.worker_seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            while True:
                ep_step += 1
                action = self.local_agent.predict_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_r += reward
                if terminated or ep_step >= self.max_steps:
                    print("Worker {} finished episode {} with reward {}".format(self.worker_id,self.global_ep.value,ep_r))
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    self.global_r_que.put(ep_r)
                    break
        
    def evaluate(self):
        sum_eval_reward = 0
        for _ in range(self.eval_eps):
            state = self.env.reset(seed = self.worker_seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            while True:
                ep_step += 1
                action = self.local_agent.predict_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_r += reward
                if terminated or ep_step >= self.max_steps:
                    break
            sum_eval_reward += ep_r
        mean_eval_reward = sum_eval_reward / self.eval_eps
        return mean_eval_reward
    def run(self):
        all_seed(self.seed)
        print("worker {} started".format(self.worker_id))
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()
class Trainer:
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(action)  # update env and return transitions under old_step_api of OpenAI Gym
            agent.memory.push((state,action,reward,terminated))
            if terminated:
                agent.update(None)
            else:
                agent.update(next_state)
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            if cfg.render:
                env.render()
            ep_step += 1
            action = agent.predict_action(state)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(action) # update env and return transitions under old_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step
if __name__ == "__main__":

    pass