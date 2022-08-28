import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import datetime
import argparse
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from common.utils import all_seed
from common.launcher import Launcher
from common.memories import PGReplay
from envs.register import register_env
from a2c_2 import A2C_2

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.critic_fc1 = nn.Linear(input_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, 1)

        self.actor_fc1 = nn.Linear(input_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        
        policy_dist = F.relu(self.actor_fc1(state))
        policy_dist = F.softmax(self.actor_fc2(policy_dist), dim=1)

        return value, policy_dist

class Main(Launcher):
    def get_args(self):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default='A2C',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
        parser.add_argument('--train_eps',default=400,type=int,help="episodes of training") 
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing") 
        parser.add_argument('--gamma',default=0.90,type=float,help="discounted factor") 
        parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon") 
        parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon") 
        parser.add_argument('--epsilon_decay',default=300,type=int,help="decay rate of epsilon") 
        parser.add_argument('--lr',default=0.1,type=float,help="learning rate")
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
        parser.add_argument('--seed',default=10,type=int,help="seed") 
        parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")   
        args = parser.parse_args()   
        default_args = {'result_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                        'model_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
        }
        args = {**vars(args),**default_args}  # type(dict)                         
        return args
    def env_agent_config(self,cfg):
        ''' create env and agent
        '''  
        register_env(cfg['env_name'])
        env = gym.make(cfg['env_name']) 
        if cfg['seed'] !=0: # set random seed
            all_seed(env,seed=cfg["seed"]) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        print(f"n_states: {n_states}, n_actions: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        models = {'ActorCritic':ActorCritic(cfg.n_states,cfg.n_actions, cfg.hidden_dim).to(cfg.device)}
        memories = {'ACMemories':PGReplay}
        agent = A2C_2(models,memories,cfg)
        return env,agent
    def train(self,cfg,env,agent):
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        for i_ep in range(cfg['train_eps']):
            ep_reward = 0  # reward per episode
            ep_step = 0 # step per episode
            state = env.reset()  # reset and obtain initial state
            while True:
                action = agent.sample_action(state)  # sample action
                next_state, reward, done, _ = env.step(action)  # update env and return transitions
                agent.update(state, action, reward, next_state, done)  # update agent
                state = next_state  # update state
                ep_reward += reward
                ep_step += 1
                if done:
                    break
            rewards.append(ep_reward)
            steps.append(ep_step)
            if (i_ep+1)%10==0:
                print(f'Episode: {i_ep+1}/{cfg["train_eps"]}, Reward: {ep_reward:.2f}, Steps:{ep_step}, Epislon: {agent.epsilon:.3f}')
        print("Finish training!")
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
    def test(self,cfg,env,agent):
        print("Start testing!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = [] # record steps for all episodes
        for i_ep in range(cfg['test_eps']):
            ep_reward = 0  # reward per episode
            ep_step = 0
            state = env.reset()  # reset and obtain initial state
            while True:
                action = agent.predict_action(state)  # predict action
                next_state, reward, done, _ = env.step(action)  
                state = next_state 
                ep_reward += reward
                ep_step += 1
                if done:
                    break
            rewards.append(ep_reward)
            steps.append(ep_step)
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}, Steps:{ep_step}, Reward: {ep_reward:.2f}")
        print("Finish testing!")
        return {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}

if __name__ == "__main__":
    main = Main()
    main.run()
   

        
    
