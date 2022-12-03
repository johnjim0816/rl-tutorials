#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-11-06 00:31:35
LastEditor: JiangJi
LastEditTime: 2022-12-03 18:16:40
Discription: parameters of MonteCarlo
'''
        
class AlgoConfig:
    def __init__(self) -> None:
        self.gamma = 0.90 # discount factor
        self.epsilon = 0.15 # epsilon greedy
        self.lr = 0.1 # learning rate