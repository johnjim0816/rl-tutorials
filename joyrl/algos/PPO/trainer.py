#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2022-12-17 19:56:24
Discription: 
Environment: 
'''

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
            agent.memory.push((state, action, reward, terminated, agent.probs, agent.log_probs))  # store transitions
            agent.update()  # update agent
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
    def collect_one_episode(self, env, agent, cfg):
        # dict of arrays
        collected = False
        ep_memory = {'state': [], 'action': [], 'reward': [], 'terminated': []}
        while not collected:
            trajectories = 0
            state = env.reset()
            ep_memory = {'state': [], 'action': [], 'reward': [], 'terminated': []}
            ep_reward = 0
            while trajectories < cfg.max_steps:
                action = agent.sample_action(state)
                new_state, reward, terminated, truncated, info = env.step(
                    action)
                ep_reward += reward
                ep_memory['state'].append(state)
                ep_memory['action'].append(action)
                ep_memory['reward'].append(reward)
                ep_memory['terminated'].append(terminated)
                state = new_state
                trajectories += 1
                if terminated:
                    if ep_reward >= cfg.min_reward:
                        collected = True
                    break
        return ep_reward, ep_memory['state'], ep_memory['action'], ep_memory['reward'], ep_memory['terminated']
   

        
    
