from os.path import dirname
import gym
from tqdm.auto import tqdm
import numpy as np
import warnings
import sys
# PPO
sys.path.append(dirname(__file__))
# joyrl
sys.path.append(dirname(dirname(dirname(__file__))))
from ppo2 import PPO
from config import AlgoConfig
from trainer import Trainer 
from common.models import ActorNormal, Critic
from common.memories import PGReplay
warnings.filterwarnings('ignore')


def ppo_test():
    env_name = 'Pendulum-v1'
    fpath = r'D:\TMP\PPOTest'
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    cfg = AlgoConfig()
    models = {
        "Actor": ActorNormal(state_dim, action_dim, hidden_dim=cfg.actor_hidden_dim),
        "Critic": Critic(state_dim, 1, hidden_dim=cfg.critic_hidden_dim)
    }
    memory = PGReplay()
    agent = PPO(models, memory, cfg)
    trHelper = Trainer()
    
    now_reward = -np.inf
    reward_list = []
    bf_reward = -np.inf
    tq_bar = tqdm(range(cfg.num_episode))
    mini_b = cfg.minibatch_size
    for i in tq_bar:
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode} ](minibatch={mini_b})')    
        agent, ep_reward, ep_step = trHelper.train_one_episode(env, agent, cfg)
        reward_list.append(ep_reward)
        now_reward = np.mean(reward_list[-10:])
        if (bf_reward < now_reward) and (i >= 10):
            agent.save_model(fpath)
            bf_reward = now_reward
        tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})

    env.close()
    # load best
    env = gym.make(env_name, render_mode='human')
    cfg.render = True
    agent.load_model(fpath)
    agent, ep_reward, ep_step = trHelper.test_one_episode(env, agent, cfg)
    print(f'Play reward={ep_reward} step={ep_step}')
    env.close()


if __name__ == "__main__":
    ppo_test()



