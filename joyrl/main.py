import sys, os

os.environ[
    "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import argparse
import yaml
from pathlib import Path
import datetime
import gym
from config.config import GeneralConfig
from common.utils import get_logger, save_results, save_cfgs, plot_rewards, merge_class_attrs, all_seed
from envs.register import register_env

class MergedConfig:
    def __init__(self) -> None:
        pass
class Main(object):
    def __init__(self) -> None:
        pass

    def get_default_cfg(self):
        self.general_cfg = GeneralConfig()
        self.algo_name = self.general_cfg.algo_name
        algo_mod = __import__(f"algos.{self.algo_name}.config", fromlist=['AlgoConfig'])
        self.algo_cfg = algo_mod.AlgoConfig()
        self.cfgs = {'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg}

    def print_cfgs(self, cfg, logger):
        ''' print parameters
        '''
        cfg_dict = vars(cfg)
        logger.info("Hyperparameters:")
        logger.info(''.join(['='] * 80))
        tplt = "{:^20}\t{:^20}\t{:^20}"
        logger.info(tplt.format("Name", "Value", "Type"))
        for k, v in cfg_dict.items():
            if v.__class__.__name__ == 'list':
                v = str(v)
            if v is None:
                v = 'None'
            logger.info(tplt.format(k, v, str(type(v))))
        logger.info(''.join(['='] * 80))

    def process_yaml_cfg(self):
        ''' load yaml config
        '''
        parser = argparse.ArgumentParser(description="hyperparameters")
        parser.add_argument('--yaml', default='presets/CartPole-v1_GAIL_Train.yaml', type=str,
                            help='the path of config file')
        args = parser.parse_args()
        if args.yaml is not None:
            with open(args.yaml) as f:
                load_cfg = yaml.load(f, Loader=yaml.FullLoader)
                # load algo config
                self.algo_name = load_cfg['general_cfg']['algo_name']
                algo_mod = __import__(f"algos.{self.algo_name}.config",
                                      fromlist=['AlgoConfig'])  # dynamic loading of modules
                self.algo_cfg = algo_mod.AlgoConfig()
                self.cfgs = {'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg}
                # merge config
                for cfg_type in self.cfgs:
                    if load_cfg[cfg_type] is not None:
                        for k, v in load_cfg[cfg_type].items():
                            setattr(self.cfgs[cfg_type], k, v)

    def create_dirs(self, cfg):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        self.task_dir = f"{curr_path}/tasks/{cfg.mode.capitalize()}_{cfg.env_name}_{cfg.algo_name}_{curr_time}"
        Path(self.task_dir).mkdir(parents=True, exist_ok=True)
        self.model_dir = f"{self.task_dir}/models"
        self.res_dir = f"{self.task_dir}/results/"
        self.log_dir = f"{self.task_dir}/logs/"
        self.traj_dir = f"{self.task_dir}/traj/"

    def env_config(self, cfg, logger):
        ''' configure environment
        '''
        register_env(cfg.env_name)
        if cfg.render:
            env = gym.make(cfg.env_name, new_step_api=cfg.new_step_api, render_mode=cfg.render_mode)  # create env
        else:
            env = gym.make(cfg.env_name, new_step_api=cfg.new_step_api)  # create env
        if cfg.wrapper is not None:
            wrapper_class_path = cfg.wrapper.split('.')[:-1]
            wrapper_class_name = cfg.wrapper.split('.')[-1]
            env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
            env = getattr(env_wapper, wrapper_class_name)(env, new_step_api=cfg.new_step_api)
        all_seed(seed=cfg.seed)  # set seed == 0 means no seed
        try:  # state dimension
            n_states = env.observation_space.n  # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0]  # print(hasattr(env.observation_space, 'shape'))
        try:
            n_actions = env.action_space.n  # action dimension
        except AttributeError:
            n_actions = env.action_space.shape[0]
            logger.info(f"action_bound: {abs(env.action_space.low.item())}")
            setattr(cfg, 'action_bound', abs(env.action_space.low.item()))
        setattr(cfg, 'action_space', env.action_space)
        logger.info(f"n_states: {n_states}, n_actions: {n_actions}")  # print info
        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        return env

    def evaluate(self, cfg, trainer, env, agent):
        sum_eval_reward = 0
        for _ in range(cfg.eval_eps):
            _, eval_ep_reward, _ = trainer.test_one_episode(env, agent, cfg)
            sum_eval_reward += eval_ep_reward
        mean_eval_reward = sum_eval_reward / cfg.eval_eps
        return mean_eval_reward

    def run(self) -> None:
        self.get_default_cfg()  # get default config
        self.process_yaml_cfg()  # process yaml config
        cfg = MergedConfig()  # merge config
        cfg = merge_class_attrs(cfg, self.cfgs['general_cfg'])
        cfg = merge_class_attrs(cfg, self.cfgs['algo_cfg'])
        self.create_dirs(cfg)  # create dirs
        logger = get_logger(self.log_dir)  # create the logger
        self.print_cfgs(cfg, logger)  # print the configuration
        env = self.env_config(cfg, logger)  # configure environment
        agent_mod = __import__(f"algos.{cfg.algo_name}.agent", fromlist=['Agent'])
        agent = agent_mod.Agent(cfg)  # create agent
        trainer_mod = __import__(f"algos.{cfg.algo_name}.trainer", fromlist=['Trainer'])
        trainer = trainer_mod.Trainer()  # create trainer
        if cfg.load_checkpoint:
            agent.load_model(f"tasks/{cfg.load_path}/models")
        logger.info(f"Start {cfg.mode}ing!")
        logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = []  # record rewards for all episodes
        steps = []  # record steps for all episodes
        if cfg.mode.lower() == 'train':
            best_ep_reward = -float('inf')
            for i_ep in range(cfg.train_eps):
                agent, ep_reward, ep_step = trainer.train_one_episode(env, agent, cfg)
                logger.info(f"Episode: {i_ep + 1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                rewards.append(ep_reward)
                steps.append(ep_step)
                # for _ in range
                if (i_ep + 1) % cfg.eval_per_episode == 0:
                    mean_eval_reward = self.evaluate(cfg, trainer, env, agent)
                    if mean_eval_reward >= best_ep_reward:  # update best reward
                        logger.info(f"Current episode {i_ep + 1} has the best eval reward: {mean_eval_reward:.3f}")
                        best_ep_reward = mean_eval_reward
                        agent.save_model(self.model_dir)  # save models with best reward
            # env.close()
        elif cfg.mode.lower() == 'test':
            for i_ep in range(cfg.test_eps):
                agent, ep_reward, ep_step = trainer.test_one_episode(env, agent, cfg)
                logger.info(f"Episode: {i_ep + 1}/{cfg.test_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                rewards.append(ep_reward)
                steps.append(ep_step)
            agent.save_model(self.model_dir)  # save models
            # env.close()
        elif cfg.mode.lower() == 'collect':  # collect
            memory = {'states': [], 'actions': [], 'rewards': [], 'terminals': []}
            for i_ep in range(cfg.collect_eps):
                total_reward, ep_state, ep_action, ep_reward, ep_terminal = trainer.collect_one_episode(env, agent, cfg)
                memory['states'] += ep_state
                memory['actions'] += ep_action
                memory['rewards'] += ep_reward
                memory['terminals'] += ep_terminal
                logger.info(f'trajectories {i_ep + 1} collected, reward {total_reward}')
                rewards.append(total_reward)
                steps.append(cfg.max_steps)
            env.close()
            agent.save_traj(memory, self.traj_dir)
            logger.info(f"trajectories saved to {self.traj_dir}")
        logger.info(f"Finish {cfg.mode}ing!")
        res_dic = {'episodes': range(len(rewards)), 'rewards': rewards, 'steps': steps}
        save_results(res_dic, self.res_dir)  # save results
        save_cfgs(self.cfgs, self.task_dir)  # save config
        plot_rewards(rewards,
                     title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}",
                     fpath=self.res_dir)


if __name__ == "__main__":
    main = Main()
    main.run()
