#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-10-07 20:57:11
LastEditor: John
LastEditTime: 2021-03-12 16:13:46
Discription: 
Environment: 
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
def plot_rewards(rewards,ma_rewards,tag="train",algo = "On-Policy First-Visit MC Control",path='./'):
    sns.set()
    plt.title("average learning curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.plot(ma_rewards,label='moving average rewards')
    plt.savefig(path+"rewards_curve_{}".format(tag))
    plt.legend()
    plt.show()
