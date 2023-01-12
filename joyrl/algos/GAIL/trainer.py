#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: Yi Zhang
LastEditTime: 2022-12-03 16:25:37
Discription: 
Environment: 
'''

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
            agent.adversarial_update(cfg)
            agent.update(cfg)
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
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
