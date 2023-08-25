
import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
util_path = os.path.dirname(parent_path)
sys.path.append(parent_path)  # add path to system path
sys.path.append(util_path)  # add path to system path

from BC.BC_trainer import BC
from BC.utils import DataSet, get_one_hot, preprocess_obs, softmax, data_collection, save_frames_as_gif
from common.launcher import Launcher
from common.utils import get_logger,merge_class_attrs
from config.config import GeneralConfigBC, AlgoConfigBC, MergedConfig

import datetime
from pathlib import Path
import gym
import numpy as np
import cv2
import torch 
from torch.utils.data import DataLoader

### in the file there are many episodes of demonstration, we can choose a few from it for learning purpose
class Main(Launcher):
    def __init__(self):
        self.cfg = MergedConfig()
        super().__init__()
        self.cfg = merge_class_attrs(self.cfg,GeneralConfigBC())
        self.cfg = merge_class_attrs(self.cfg,AlgoConfigBC())
        self.create_path(self.cfg)
        self.logger = get_logger(self.log_dir)
        self.env_agent_config(self.logger)
        
    def create_path(self,cfg):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # obtain current time
        self.task_dir = f"tasks/{cfg.env_name}_{cfg.algo_name}_{curr_time}"
        Path(self.task_dir).mkdir(parents=True, exist_ok=True)
        self.model_dir = f"{self.task_dir}/models/"
        if not os.path.exists(f'./data/{self.cfg.env_name}'):
            os.makedirs(f'./data/{self.cfg.env_name}')
        self.data_dir = f'./data/{self.cfg.env_name}/'
        self.log_dir = f"{self.task_dir}/logs/"

    def env_agent_config(self,logger):
        ''' create env and agent
        '''
        env = gym.make(self.cfg.env_name)  # create env
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        try:
            n_actions = env.action_space.n  # action dimension
        except AttributeError:
            n_actions = env.action_space.shape[0]
            logger.info(f"action_bound: {abs(env.action_space.low.item())}") 
            setattr(self.cfg, 'action_bound', abs(env.action_space.low.item()))
        logger.info(f"n_states: {n_states}, n_actions: {n_actions}") # print info
        # update to cfg paramters
        setattr(self.cfg, 'n_states', n_states)
        setattr(self.cfg, 'n_actions', n_actions)
        # create agent using cfgs
        self.bc_agent = BC(self.cfg, self.model_dir)
        logger.info(f"Network Architecture: {self.bc_agent.bc_network.network}")
        
    def train(self):
        # load data
        try:
            data = np.load(f'{self.data_dir}/expert_data.npz')
        except:
            self.logger.info("Demonstration data unavailable, start training a PPO agent for data collection")
            data_collection(self.cfg.env_name, self.data_dir)
            data = np.load(f'{self.data_dir}/expert_data.npz')

        total_states, total_actions = data['expert_observations'], data['expert_actions']
        # encode actions to onehot encoder if the action space is discrete
        if not self.cfg.continuous_action:
            total_actions = get_one_hot(np.array([total_actions]).reshape(-1), self.cfg.n_actions)
        # preprocess image data
        if self.cfg.image_obs:
            total_states = preprocess_obs(total_states, self.image_shape)
        # train the network
        data = DataSet(total_states, total_actions)
        train_dataloader = DataLoader(data, batch_size=self.cfg.batch_size, shuffle=True)
        self.logger.info("Data loaded, start training")
        self.bc_agent.train(train_dataloader)
        self.logger.info("Finished Training")
    def test(self, model_dir, model_idx=99):
        env = gym.make(self.cfg.env_name)
        # we can set the maximum steps here
        env._max_episode_steps = self.cfg.max_steps
        frames = []
        state = np.array([env.reset()])
        bc = BC(self.cfg, model_dir)
        bc.load_policy(model_idx)
        bc.bc_network.eval()
        reward = 0
        while True:
            if self.cfg.render:
                # env.render()
                frame = env.render(mode="rgb_array")
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                frame = cv2.putText(img=np.array(frame), text=f'reward:{str(reward)}', org=(10, 30),fontFace=font, fontScale=1, thickness=1, color=color)
                frames.append(frame)

            if self.cfg.image_obs:
                state = preprocess_obs(state).float().clone().detach().to(self.cfg.device)
            state = torch.from_numpy(state).float().clone().detach().to(self.cfg.device)
            action = bc.bc_network.forward(state).cpu().detach().numpy()

            if not self.cfg.continuous_action:   
                action = np.argmax(softmax(action))

            next_state, r, terminated, _ = env.step(action)
            reward += r
            state = np.array([next_state])
            if terminated:
                break
        env.close()
        self.logger.info(f'The final reward is: {reward}')
        save_frames_as_gif(frames, path = './materials/')

    def run(self):
        if self.cfg.mode == 'train':
            self.train()
        elif self.cfg.mode == 'test':
            self.test(self.cfg.model_path, 199)


if __name__ == '__main__':
    main = Main()
    main.run()
