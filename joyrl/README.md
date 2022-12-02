[中文](./README_cn.md)|EN

## JoyRL

## Install

Currently JoyRL support Python3.7 and Gym0.25.2

```bash
conda create -n joyrl python=3.7
conda activate joyrl
pip install -r requirements.txt
```
Torch:

```bash
# CPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# GPU with mirrors
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Usage

you can simply change the parameters (like env_name, algo_name) in `config.config.GeneralConfig()` and run:
```bash
python main.py
```
then it will a new folder named `tasks` to save results and models.

Or you can custom parameters with a `yaml` file as you can seen in  `config/custom_config_Train.yaml` and run:
```bash
python main.py --yaml config/custom_config_Train.yaml
```
And there are presets yaml files in the [presets](./presets/) folder and well trained results in the [benchmarks](./benchmarks/) folder.

## Docs

please click [docs](./docs/README.md)

## Environments

Please click [envs](./envs/README.md) to read environments instruments.

## Algorithms

|       Name       |                          Reference                           |                    Author                     | Notes |
| :--------------: | :----------------------------------------------------------: | :-------------------------------------------: | :---: |
| [Value Iteration](./algos/VI/) | [RL introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) | [guoshicheng](https://github.com/gsc579) |  |
| [DQN](./algos/DQN/) | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [johnjim0816](https://github.com/johnjim0816) |       |
| [PER_DQN](./algos/PER_DQN/) | [PER_DQN Paper](https://arxiv.org/pdf/1511.05952) | [wangzhongren](https://github.com/wangzhongren-code) |       |
| [NoisyDQN](./algos/NoisyDQN/) | [NoisyDQN Paper](https://arxiv.org/pdf/1706.10295.pdf) | [wangzhongren](https://github.com/wangzhongren-code) |       |
| [DDPG](./algos/DDPG/) | [DDPG Paper](https://arxiv.org/abs/1509.02971) | [johnjim0816](https://github.com/johnjim0816) |       |