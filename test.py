#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-06-11 09:33:10
LastEditor: John
LastEditTime: 2020-08-20 11:54:06
Discription: 
Environment: 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/1/11 19:02
# @Author  : Arrow and Bullet
# @FileName: min_max.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_41800366

from numpy import array  # 从numpy中引入array，为创建矩阵做准备


A = array([[1, 2, 3],  # 创建一个4行3列的矩阵
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12]])

B = A.min(0)  # 返回A每一列最小值组成的一维数组；
print(B)  # 结果 ：[1 2 3]

B = A.min(1)  # 返回A每一行最小值组成的一维数组；
print(B)  # 结果 ：[ 1  4  7 10]

B = A.max(0)  # 返回A每一列最大值组成的一维数组；
print(B)  # 结果 ：[10 11 12]

B = A.max(1)  # 返回A每一行最大值组成的一维数组；
print(B)  # 结果 ：[ 3  6  9 12]
