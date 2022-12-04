#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-12-03 19:40:32
LastEditor: JiangJi
LastEditTime: 2022-12-04 14:22:06
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
from agent import Agent
from config import AlgoConfig
from common.utils import all_seed
from common.models import ActorSoftmax, Critic

class Worker(mp.Process):
    def __init__(self,cfg,env,agent,worker_id,global_ep,global_ep_r,res_queue):
        super(Worker,self).__init__()
        self.worker_id = worker_id
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        self.env = env
        self.agent = agent
        self.seed = cfg.seed + worker_id
        self.train_eps = cfg.train_eps
        self.max_steps = cfg.max_steps
    def run(self):
        print("worker {} started".format(self.worker_id))
        all_seed(self.seed)
        while self.global_ep.value <= self.train_eps:
            state = self.env.reset(seed = self.seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            for _ in range(self.max_steps):
                ep_step += 1
                action = self.agent.sample_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.agent.memory.push((state, action, reward, terminated))
                self.agent.update(next_state)
                state = next_state
                ep_r += reward
                if terminated:
                    print("worker {} finished episode {} with reward {}".format(self.worker_id,self.global_ep.value,ep_r))
                    with self.global_ep_r.get_lock():
                        self.global_ep_r.value = ep_r
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    self.res_queue.put(ep_r)
                    break
if __name__ == "__main__":
    
    cfg = AlgoConfig()
    env = gym.make(cfg.env_name,new_step_api=True)
    # all_seed(env,seed = cfg.seed) # set seed == 0 means no seed
    try: # state dimension
        n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
    except AttributeError:
        n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
    try:
        n_actions = env.action_space.n  # action dimension
    except AttributeError:
        n_actions = env.action_space.shape[0]
        setattr(cfg, 'action_bound', abs(env.action_space.low.item()))
    setattr(cfg, 'action_space', env.action_space)
    # update to cfg paramters
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions)
    # mp.set_start_method("spawn")
    
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()
    share_agent = Agent(cfg, share_agent = None)
    agent = Agent(cfg, share_agent = share_agent)
    workers = [Worker(cfg,env,agent,worker_id,global_ep,global_ep_r,res_queue) for worker_id in range(cfg.n_workers)]
    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()
    print("training finished")
    res = [] # record episode reward to plot
    while not res_queue.empty():
        r = res_queue.get()
        res.append(r)
    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
    

