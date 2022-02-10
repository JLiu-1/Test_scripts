import numpy as np 

import os
import argparse
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import math
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
parser.add_argument('--step','-s',type=int,default=4)
parser.add_argument('--rate','-r',type=float,default=1.0)
parser.add_argument('--level_rate','-lr',type=float,default=1.0)
parser.add_argument('--anchor','-a',type=int,default=-1)
#parser.add_argument('--norm_max','-nx',type=float,default=1)
#parser.add_argument('--norm_min','-ni',type=float,default=-1)
#parser.add_argument('--max','-mx',type=float,default=1)
#parser.add_argument('--min','-mi',type=float,default=0)
#parser.add_argument('--level','-l',type=int,default=2)
#parser.add_argument('--noise','-n',type=bool,default=False)
#parser.add_argument('--intercept','-t',type=bool,default=False)
args = parser.parse_args()
#level=args.level

#actv_dict={"no":nn.Identity,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh}
#actv=actv_dict[args.actv]
#model=nn.Sequential(nn.Linear(7,1),actv())
#model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
#model.eval()
#for name,parameters in model.named_parameters():
#    print(name)
#    print(parameters.detach().numpy())
size_x=512
size_y=512
size_z=512
array=np.fromfile(args.input,dtype=np.float32).reshape((size_x,size_y,size_z))
#coefs=np.fromfile(args.checkpoint,dtype=np.float64)
error_bound=args.error*(np.max(array)-np.min(array))
step=args.step
rate=args.rate
rated_error_bound=error_bound/rate
lr=args.level_rate
def get_block_index(x,block):
    
    return x//block

#block_size=args.block
#blocked_size_x=(size_x-1)//block_size+1
#blocked_size_y=(size_y-1)//block_size+1
#size=level**2-1
#coef_array=np.zeros((blocked_size_x,blocked_size_y,size),dtype=np.double)

#intercept_array=np.zeros((blocked_size_x,blocked_size_y),dtype=np.double)
qs=[]
max_level=int(math.log(step,2))

us=[]
lorenzo_qs=[]
'''
for x_idx,x_start in enumerate(range(0,size_x,block_size)):
    for y_idx,y_start in enumerate(range(0,size_y,block_size)): 
        
        #x_end=min(x_start+block_size,size_x)
        #y_end=min(y_start+block_size,size_y)

        if block_size%2==0:
            #x_end-=1
            #y_end-=1
            compli=True
        else:
            compli=False
        
        for x in range(x_start,x_end,2):
            for y in range(y_start,y_end,2):
                f_01=array[x-2][y] if x>=2 else 0
                f_10=array[x][y-2] if y>=2 else 0
            
                f_00=array[x-2][y-2] if x>=2 and y>=2 else 0
                
                pred=f_01+f_10-f_00
                q,decomp=quantize(orig,pred,error_bound)
                qs_array[x-x_start][y-y_start]=q
                if 
                   
                
        reg_xs=np.array(reg_xs).astype(np.double)
        reg_ys=np.array(reg_ys).astype(np.double)
        res=LinearRegression(fit_intercept=args.intercept).fit(reg_xs, reg_ys)
        coef_array[x_idx][y_idx]=res.coef_
        intercept_array[x_idx][y_idx]=res.intercept_

print(coef_array[0][0].shape)

'''
anchor=args.anchor
for x in range(0,size_x,step):
    for y in range(0,size_y,step):
        for z in range(0,size_z,step):
            if anchor>0 and x%anchor==0 and y%anchor==0 and z%anchor==0:
                continue
            orig=array[x][y][z]
            f_011=array[x-step][y][z] if x else 0
            f_101=array[x][y-step][z] if y else 0
            f_110=array[x][y][z-step] if z else 0
            f_001=array[x-step][y-step][z] if x and y else 0
            f_100=array[x][y-step][z-step] if y and z else 0
            f_010=array[x-step][y][z-step] if x and z else 0
            f_000=array[x-step][y-step][z-step] if x and y and z else 0
                
            pred=f_000+f_011+f_101+f_110-f_001-f_010-f_100
                
        
                
            q,decomp=quantize(orig,pred,rated_error_bound)
            qs.append(q)
            if q==0:
                us.append(decomp)
            array[x][y][z]=decomp

def interp(array,level,eb):#only 2^n+1 cubic array
    if level==max_level:
        return 
    print(array.shape)
    side_length_x=array.shape[0]
    side_length_y=array.shape[1]
    side_length_z=array.shape[2]
    sparse_grid=array[0:side_length_x:2,0:side_length_y:2,0:side_length_z:2]
    '''
    cur_eb=error_bound#/(2**level)
    if cur_eb<error_bound/10:
        cur_eb=error_bound/10
    '''
    interp(sparse_grid,level+1,eb/lr)
    
      
    
    for x in range(0,side_length_x,2):
        for y in range(0,side_length_y,2):
            for z in (1,side_length_z,2):
                if z==side_length_z-1:
                    continue
                orig=array[x][y][z]
                pred=(array[x][y][z-1]+array[x][y][z+1])/2
                q,decomp=quantize(orig,pred,eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp


    for x in range(0,side_length_x,2):
        for y in range(1,side_length_y,2):
            for z in (0,side_length_z,2):
                if y==side_length_y-1:
                    continue
                orig=array[x][y][z]
                pred=(array[x][y-1][z]+array[x][y+1][z])/2
                q,decomp=quantize(orig,pred,eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp 
    for x in range(1,side_length_x,2):
        for y in range(0,side_length_y,2):
            for z in (0,side_length_z,2):
                if x==side_length_x-1:
                    continue
                orig=array[x][y][z]
                pred=(array[x-1][y][z]+array[x+1][y][z])/2
                q,decomp=quantize(orig,pred,eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp 
    
    for x in range(1,side_length_x,2):
        for y in range(1,side_length_y,2):
            for z in range(0,side_length_z,2):
                if x==side_length_x-1 or y==side_length_y-1:
                    continue
                orig=array[x][y][z]
                pred=(array[x-1][y][z]+array[x+1][y][z]+array[x][y-1][z]+array[x][y+1][z])/4
                q,decomp=quantize(orig,pred,eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp
    for x in range(1,side_length_x,2):
        for y in range(0,side_length_y,2):
            for z in range(1,side_length_z,2):
                if x==side_length_x-1 or z==side_length_z-1:
                    continue
                orig=array[x][y][z]
                pred=(array[x-1][y][z]+array[x+1][y][z]+array[x][y][z-1]+array[x][y][z+1])/4
                q,decomp=quantize(orig,pred,eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp
    for x in range(0,side_length_x,2):
        for y in range(1,side_length_y,2):
            for z in range(1,side_length_z,2):
                if y==side_length_y-1 or z==side_length_z-1:
                    continue
                orig=array[x][y][z]
                pred=(array[x][y-1][z]+array[x][y+1][z]+array[x][y][z-1]+array[x][y][z+1])/4
                q,decomp=quantize(orig,pred,eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp
    for x in range(1,side_length_x,2):
        for y in range(1,side_length_y,2):
            for z in range(1,side_length_z,2):
                if x==side_length_x-1 or y==side_length_y-1 or z==side_length_z-1:
                    continue
                orig=array[x][y][z]
                pred=(array[x-1][y][z]+array[x+1][y][z]+array[x][y-1][z]+array[x][y+1][z]+array[x][y][z-1]+array[x][y][z+1])/6
                q,decomp=quantize(orig,pred,eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp


last_x=((size_x-1)//step)*step
last_y=((size_y-1)//step)*step
last_z=((size_z-1)//step)*step
interp(array[:last_x+1,:last_y+1,:last_z+1],0,error_bound)
def lorenzo_3d(array,x_start,x_end,y_start,y_end,z_start,z_end):
    for x in range(x_start,x_end):
        for y in range(y_start,y_end):
            for z in range(z_start,z_end):
                if x<=last_x and y<=last_y and z<=last_z:
                    continue
                orig=array[x][y][z]
        
                f_011=array[x-1][y][z] if x else 0
                f_101=array[x][y-1][z] if y else 0
                f_110=array[x][y][z-1] if z else 0
                f_001=array[x-1][y-1][z] if x and y else 0
                f_100=array[x][y-1][z-1] if y and z else 0
                f_010=array[x-1][y][z-1] if x and z else 0
                f_000=array[x-1][y-1][z-1] if x and y and z else 0
                
                pred=f_000+f_011+f_101+f_110-f_001-f_010-f_100
                
        
                
                q,decomp=quantize(orig,pred,error_bound)
                lorenzo_qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y][z]=decomp
lorenzo_3d(array,0,size_x,0,size_y,0,size_z)






quants=np.concatenate( (np.array(lorenzo_qs,dtype=np.int32),np.array(qs,dtype=np.int32) ) )
unpreds=np.array(us,dtype=np.float32)
array.tofile(args.output)
quants.tofile("nyx_q.dat")
unpreds.tofile("nyx_u.dat")