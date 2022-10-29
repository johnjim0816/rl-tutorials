import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import gym
import torch
import numpy as np
from common.utils import all_seed,merge_class_attrs
from common.launcher import Launcher
from common.memories import PGReplay
from common.models import ActorSoftmax,Critic
from envs.register import register_env
from a2c import A2C
from config.config import GeneralConfigA2C,AlgoConfigA2C

class Main(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.cfgs['general_cfg'] = merge_class_attrs(self.cfgs['general_cfg'],GeneralConfigA2C())
        self.cfgs['algo_cfg'] = merge_class_attrs(self.cfgs['algo_cfg'],AlgoConfigA2C())
    def env_agent_config(self,cfg,logger):
        ''' create env and agent
        '''  
        register_env(cfg.env_name)
        env = gym.make(cfg.env_name,new_step_api=True)  # create env
        if cfg.seed !=0: # set random seed
            all_seed(env,seed = cfg.seed) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        logger.info(f"n_states: {n_states}, n_actions: {n_actions}") # print info
        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        models = {'Actor':ActorSoftmax(n_states,n_actions, hidden_dim = cfg.actor_hidden_dim),'Critic':Critic(n_states,1,hidden_dim=cfg.critic_hidden_dim)}
        memories = {'ACMemory':PGReplay()}
        agent = A2C(models,memories,cfg)
        return env,agent
    def train(self,cfg,env,agent,logger):
        logger.info("Start training!")
        logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        for i_ep in range(cfg.train_eps):
            ep_reward = 0  # reward per episode
            ep_step = 0 # step per episode
            ep_entropy = 0
            state = env.reset()  # reset and obtain initial state
            for _ in range(cfg.max_steps):
                action, value, dist = agent.sample_action(state)  # sample action
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions
                log_prob = torch.log(dist.squeeze(0)[action])
                entropy = - np.sum(np.mean(dist.detach().cpu().numpy()) * np.log(dist.detach().cpu().numpy()))
                agent.memory.push((value,log_prob,reward))  # save transitions
                state = next_state  # update state
                ep_reward += reward
                ep_entropy += entropy
                ep_step += 1
                if terminated:
                    break
            agent.update(next_state,ep_entropy)  # update agent
            rewards.append(ep_reward)
            steps.append(ep_step)
            logger.info(f"Episode: {i_ep+1}/{cfg.train_eps}, Reward: {ep_reward:.2f}, Steps:{ep_step}")
        logger.info("Finish training!")
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
    def test(self,cfg,env,agent,logger):
        logger.info("Start testing!")
        logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        for i_ep in range(cfg.test_eps):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            for _ in range(cfg.max_steps):
                action,_,_ = agent.predict_action(state)  # predict action
                next_state, reward, terminated, truncated , info = env.step(action)  
                state = next_state 
                ep_reward += reward
                ep_step += 1
                if terminated:
                    break
            rewards.append(ep_reward)
            steps.append(ep_step)
            logger.info(f"Episode: {i_ep+1}/{cfg.test_eps}, Reward: {ep_reward:.2f}, Steps:{ep_step}")
        logger.info("Finish testing!")
        env.close()
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}

if __name__ == "__main__":
    main = Main()
    main.run()
   

        
    
