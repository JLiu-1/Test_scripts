import numpy as np 

import os
import argparse
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
def quantize(data,pred,error_bound):
    radius=32768
    
    diff = data - pred
    quant_index = (int) (abs(diff)/ error_bound) + 1
    #print(quant_index)
    if (quant_index < radius * 2) :
        quant_index =quant_index>> 1
        half_index = quant_index
        quant_index =quant_index<< 1
        #print(quant_index)
        quant_index_shifted=0
        if (diff < 0) :
            quant_index = -quant_index
            quant_index_shifted = radius - half_index
        else :
            quant_index_shifted = radius + half_index
        
        decompressed_data = pred + quant_index * error_bound
        #print(decompressed_data)
        if abs(decompressed_data - data) > error_bound :
            #print("b")
            return 0,data
        else:
            #print("c")
            data = decompressed_data
            return quant_index_shifted,data
        
    else:
        #print("a")
        return 0,data

parser = argparse.ArgumentParser()
#parser.add_argument('--dropout','-d',type=float,default=0)
parser.add_argument('--error','-e',type=float,default=1e-3)
parser.add_argument('--input','-i',type=str)
parser.add_argument('--output','-o',type=str)
#parser.add_argument('--actv','-a',type=str,default='tanh')
parser.add_argument('--checkpoint','-c',type=str,default=None)
parser.add_argument('--block','-b',type=int,default=64)
#parser.add_argument('--norm_max','-nx',type=float,default=1)
#parser.add_argument('--norm_min','-ni',type=float,default=-1)
parser.add_argument('--max','-mx',type=float,
                   default=1)
parser.add_argument('--min','-mi',type=float,
                   default=0)
parser.add_argument('--level','-l',type=int,
                   default=2)

args = parser.parse_args()
level=args.level
#actv_dict={"no":nn.Identity,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh}
#actv=actv_dict[args.actv]
#model=nn.Sequential(nn.Linear(7,1),actv())
#model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
#model.eval()
#for name,parameters in model.named_parameters():
#    print(name)
#    print(parameters.detach().numpy())
size_x=1800
size_y=3600
array=np.fromfile(args.input,dtype=np.float32).reshape((size_x,size_y))
#coefs=np.fromfile(args.checkpoint,dtype=np.float64)
error_bound=args.error*(np.max(array)-np.min(array))

def get_block_index(x,block):
    
    return x//block

block_size=args.block
blocked_size_x=(size_x-1)//block_size+1
blocked_size_y=(size_y-1)//block_size+1
size=2*(level**2-1)
coef_array=np.zeros((blocked_size_x,blocked_size_y,size),dtype=np.double)
qs=[]
us=[]

for x_idx,x_start in enumerate(range(0,size_x,block_size)):
    for y_idx,y_start in enumerate(range(0,size_y,block_size)): 
        reg_xs=[]
        reg_ys=[]
        x_end=min(x_start+block_size,size_x)
        y_end=min(y_start+block_size,size_y)
        for x in range(x_start,x_end):
            for y in range(y_start,y_end):
                if not (x>=level-1 and y>=level-1):
                    continue
                block=array[x-level+1:x+1,y-level+1:y+1].flatten()
                reg_xs.append( np.concatenate((block[:size//2],block[:size//2]**2)) )
                reg_ys.append(block[size])
        reg_xs=np.array(reg_xs).astype(np.double)
        reg_ys=np.array(reg_ys).astype(np.double)
        coef_array[x_idx][y_idx]=LinearRegression(fit_intercept=False).fit(reg_xs, reg_ys).coef_

print(coef_array[0][0].shape)


for x in range(size_x):
    for y in range(size_y):
        
        orig=array[x][y]
        if not (x>=level-1 and y>=level-1):
            f_01=array[x-1][y] if x else 0
            f_10=array[x][y-1] if y else 0
            
            f_00=array[x-1][y-1] if x and y else 0
                
            pred=f_01+f_10-f_00
                
        else:
            block=array[x-level+1:x+1,y-level+1:y+1].flatten()[:size]
            block=np.concatenate((block,block**2))
            blockid_x=get_block_index(x,block_size)
            blockid_y=get_block_index(y,block_size)
            coefs=coef_array[blockid_x][blockid_y]
            pred=np.sum(block*coefs)
                
        q,decomp=quantize(orig,pred,error_bound)
        qs.append(q)
        if q==0:
            us.append(decomp)
        array[x][y]=decomp

quants=np.array(qs,dtype=np.int32)
unpreds=np.array(us,dtype=np.float32)
array.tofile(args.output)
quants.tofile("cld_q.dat")
unpreds.tofile("cld_u.dat")