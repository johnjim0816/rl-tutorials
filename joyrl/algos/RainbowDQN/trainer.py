import torch

class Trainer:
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps): ## 
            sum_reward = 0 ## N-step rewards
            # print ("cfg.n_step = ", cfg.n_step)
            for j in range(cfg.n_step):
                action = agent.sample_action(state)  # sample action
                next_state, reward, terminated, truncated , info = env.step(action) # update env and return transitions under new_step_api of OpenAI Gym
                sum_reward += reward
                ep_step += 1
                 
                if j == 0 :
                    init_state = state
                
                if truncated or ep_step >= cfg.max_steps:
                    break  

            policy_val = agent.policy_net(torch.tensor(init_state, device = cfg.device))[action]
            target_val = agent.target_net(torch.tensor(next_state, device = cfg.device))
            if terminated: ## this is for the PER_DQN 
                error = abs(policy_val - sum_reward)
            else:
                error = abs(policy_val - sum_reward - (cfg.gamma ** cfg.n_step) * torch.max(target_val))

            agent.memory.push(error.cpu().detach().numpy(), (init_state, action, sum_reward,
                            next_state, terminated))  # save transitions
            agent.update()  # update agent
            state = next_state  # update next state for env
            ep_reward += sum_reward  #
            if terminated or ep_step >= cfg.max_steps:
                break
        return agent,ep_reward,ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, _ , _ = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step