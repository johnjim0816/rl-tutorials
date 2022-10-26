from common.utils import get_logger,save_results,save_cfgs,plot_rewards
import time
from pathlib import Path
import datetime

class Launcher:
    def __init__(self) -> None:
        pass
    def get_cfg(self):
        cfgs = {}
        return cfgs
    def print_cfg(self,cfg):
        cfg_dict = vars(cfg)
        print("Hyperparameters:")
        print(''.join(['=']*80))
        tplt = "{:^20}\t{:^20}\t{:^20}"
        print(tplt.format("Name", "Value", "Type"))
        for k,v in cfg_dict.items():
            print(tplt.format(k,v,str(type(v))))   
        print(''.join(['=']*80))
    def env_agent_config(self,cfg,logger):
        env,agent = None,None
        return env,agent
    def train(self,cfg, env, agent,logger):
        res_dic = {}
        return res_dic
    def test(self,cfg, env, agent,logger):
        res_dic = {}
        return res_dic
    def create_path(self,cfg,mode):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")   # obtain current time
        self.task_dir = f"{mode.capitalize()}_{cfg.env_name}_{cfg.algo_name}_{curr_time}"
        Path(self.task_dir).mkdir(parents=True, exist_ok=True)
        self.res_dir = f"{self.task_dir}/results/"
    def run(self,mode='train'):
        if mode.lower() == 'train':
            cfgs = self.get_cfg() # obtain the configuration
            cfg = cfgs['cfg']
            self.print_cfg(cfg) # print the configuration
            self.create_path(cfg,mode) # create the path to save the results
            logger = get_logger(f"{self.task_dir}/logs/")
            env, agent = self.env_agent_config(cfg,logger)
            res_dic = self.train(cfg, env, agent,logger)
            self.res_dir = f"{self.task_dir}/results/"
            save_results(res_dic, self.res_dir) # save results
            save_cfgs({'general_cfg':cfgs['general_cfg'],'algo_cfg':cfgs['algo_cfg']}, self.task_dir) # save config
            agent.save_model(path = f"{self.task_dir}/models")  # save models
            plot_rewards(res_dic['rewards'], title=f"{mode}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}" ,fpath= self.res_dir)
            
        # cfg = self.get_cfg() # obtain the configuration
        # self.print_cfg(cfg) # print the configuration

        # env, agent = self.env_agent_config(cfg)
        # res_dic = self.train(cfg, env, agent)
        # save_args(cfg,path = cfg['result_path']) # save parameters
        # agent.save_model(path = cfg['model_path'])  # save models
        # save_results(res_dic, tag = 'train', path = cfg['result_path']) # save results
        # plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "train")  # plot results
        # # testing
        # # env, agent = self.env_agent_config(cfg) # create new env for testing, sometimes can ignore this step
        # agent.load_model(path = cfg['model_path'])  # load model
        # res_dic = self.test(cfg, env, agent)
        # save_results(res_dic, tag='test',
        #             path = cfg['result_path'])  
        # plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "test") 
