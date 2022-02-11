import numpy as np 

import os
import argparse
#import torch
#import torch.nn as nn
from sklearn.linear_model import LinearRegression
import math

size_x=1800
size_y=3600
array=np.fromfile("../JinyangLiu/SDRBENCH-CESM-ATM-cleared-1800x3600/CLDHGH_1_1800_3600.dat",dtype=np.float32).reshape((1800,3600))
reg_xs=[]
reg_ys=[]

for x in range(2,size_x,2):
    for y in range(1,size_y,2):
        
        if x+2>=size_x or y==size_y-1:
            continue
        orig=array[x][y]
        reg_xs.append(np.array([array[x-2][y-1],array[x-2][y+1],array[x][y-1],array[x][y+1],array[x+2][y-1],array[x+2][y+1]],dtype=np.float64))
        reg_ys.append(orig)

for x in range(1,size_x,2):
    for y in range(2,size_y,2):
        if y+2>=size_y or x==size_x-1:
            continue
        orig=array[x][y]
        orig=array[x][y]
        reg_xs.append(np.array([array[x-1][y-2],array[x+1][y-2],array[x-1][y],array[x+1][y],array[x-1][y+2],array[x+1][y+2]],dtype=np.float64))
        reg_ys.append(orig)


res=LinearRegression(fit_intercept=False).fit(reg_xs, reg_ys)
print(res.coef_)