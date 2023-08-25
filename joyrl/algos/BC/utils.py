
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from tqdm import tqdm
import gym 
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation

def data_collection(env_id, data_dir):

    env = gym.make(env_id)

    ppo_expert = PPO('MlpPolicy', env_id, verbose=1, create_eval_env=True)
    ppo_expert.learn(total_timesteps=3e4, eval_freq=10000)
    # ppo_expert.save("ppo_expert")

    mean_reward, std_reward = evaluate_policy(ppo_expert, env, n_eval_episodes=10)
    print(f"Mean reward = {mean_reward} +/- {std_reward}")

    num_interactions = int(4e4)
    expert_observations = []
    expert_actions = []
    
    obs = env.reset()

    for i in tqdm(range(num_interactions)):
        action, _ = ppo_expert.predict(obs, deterministic=True)
        expert_observations.append(obs)
        expert_actions.append(action)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    print(np.array(expert_observations).shape, np.array(expert_actions).shape)

    np.savez_compressed(
        f"{data_dir}/expert_data",
        expert_actions=expert_actions,
        expert_observations=expert_observations,
    )

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)
    
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def preprocess_obs(obs, image_shape):
    obs = np.array([cv2.cvtColor(cv2.resize(img, image_shape), cv2.COLOR_BGR2GRAY) for img in obs])
    x, w, h= obs.shape
    obs = obs.reshape((x, 1, w, h))# channel is 1 since we are using gray images
    return torch.from_numpy(obs)


class DataSet:
  
    def __init__(self, obs, label):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """
        self.observation = obs 
        print('dataset image shape is ',self.observation.shape)
        self.labels = label
        print('dataset label shape is ',self.labels.shape)

    def __len__(self):
        """return number of points in our dataset"""

        return len(self.observation)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        obs = self.observation[idx]
        label = self.labels[idx]

        return obs, label



def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=20)