#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2022-12-03 16:25:37
Discription: 
Environment: 
'''
import torch

from joyrl.algos.GAIL.dataset import TrajDataset
from tqdm import tqdm
from joyrl.algos.GAIL.utils import adversarial_imitation_update


class Trainer:
    def __init__(self) -> None:
        pass

    def train_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed=cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated, info = env.step(
                    action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(
                    action)  # update env and return transitions under old_step_api of OpenAI Gym
            agent.memory.push((state, action, agent.log_probs, reward, terminated))  # store transitions

            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                state = env.reset(seed=cfg.seed)
                if len(agent.memory) >= cfg.batch_size:
                    # train Discriminator
                    old_states, old_actions, old_log_probs, old_rewards, old_dones = agent.memory.sample()
                    # 根据agent来传入训练
                    policy_trajectory_replays = {'states': old_states, 'actions': old_actions, 'rewards': old_rewards,
                                                 'terminals': old_dones}
                    policy_trajectory = TrajDataset(policy_trajectory_replays)
                    for _ in tqdm(range(cfg.imitation_epochs), leave=False):
                        adversarial_imitation_update(agent.discriminator, agent.expert_trajectories,
                                                     policy_trajectory,
                                                     agent.discriminator_optimiser, cfg)
                    # predict reward using Discriminator
                    states = policy_trajectory_replays['states']
                    actions = policy_trajectory_replays['actions']
                    # next_states = torch.cat([policy_trajectories['states'][1:], next_state])
                    # terminals = policy_trajectories['terminals']
                    # with torch.no_grad():
                    #     policy_trajectory_replays['rewards'] = agent.discriminator.predict_reward(states, actions)
                    # update agent
                    # for _ in tqdm(range(cfg.ppo_epochs)):
                    agent.update(states, actions)
        return agent, ep_reward, ep_step

    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed=cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            if cfg.render:
                env.render()
            ep_step += 1
            action = agent.predict_action(state)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated, info = env.step(
                    action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(
                    action)  # update env and return transitions under old_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent, ep_reward, ep_step
