import numpy as np 

import os
import argparse
#import torch
#import torch.nn as nn
from sklearn.linear_model import LinearRegression
import math
import random
from multilevel_selective_compress_2d_api_rebuild import msc2d
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

parser.add_argument('--error','-e',type=float,default=1e-3)
parser.add_argument('--input','-i',type=str)
parser.add_argument('--output','-o',type=str)
parser.add_argument('--quant','-q',type=str,default="ml2_q.dat")
parser.add_argument('--unpred','-u',type=str,default="ml2_u.dat")
parser.add_argument('--max_step','-s',type=int,default=-1)
parser.add_argument('--min_coeff_level','-cl',type=int,default=99)
parser.add_argument('--rate','-r',type=float,default=1.0)
parser.add_argument('--rlist',type=float,default=-1,nargs="+")
parser.add_argument('--maximum_rate','-m',type=float,default=10.0)
#parser.add_argument('--cubic','-c',type=int,default=1)
parser.add_argument('--multidim_level','-d',type=int,default=99)
parser.add_argument('--lorenzo_fallback_check','-l',type=int,default=0)
parser.add_argument('--fallback_sample_ratio','-p',type=float,default=0.05)
parser.add_argument('--anchor_rate','-a',type=float,default=0.0)

parser.add_argument('--size_x','-x',type=int,default=1800)
parser.add_argument('--size_y','-y',type=int,default=3600)
parser.add_argument('--sz_interp','-n',type=int,default=0)
parser.add_argument('--fix','-f',type=str,default="none")
parser.add_argument('--order',type=str,default="block")
args = parser.parse_args()

size_x=args.size_x
size_y=args.size_y
array=np.fromfile(args.input,dtype=np.float32).reshape((size_x,size_y))
orig_array=np.copy(array)
#if args.lorenzo_fallback_check:
    #orig_array=np.copy(array)
predicted=np.zeros((size_x,size_y),dtype=np.int16)
rng=(np.max(array)-np.min(array))
error_bound=args.error*rng
max_step=args.max_step
rate=args.rate
maximum_rate=args.maximum_rate
anchor_rate=args.anchor_rate
max_level=int(math.log(max_step,2))
rate_list=args.rlist
if ((isinstance(rate_list,int) or isinstance(rate_list,float)) and  rate_list>0) or (isinstance(rate_list,list ) and rate_list[0]>0):

    if isinstance(rate_list,int) or isinstance(rate_list,float):
        rate_list=[rate_list]

    while len(rate_list)<max_level:
        rate_list.insert(0,rate_list[0])
else:
    rate_list=None

qs=[[] for i in range(max_level+1)]

us=[]
lorenzo_qs=[]
min_coeff_level=args.min_coeff_level
#anchor=args.anchor
'''
if max_step>0:
    
    anchor_rate=args.anchor_rate
    if anchor_rate>0:
        anchor_eb=error_bound/anchor_rate
        print("Anchor eb:%f" % anchor_eb)

        if max_level>=min_coeff_level:
            reg_xs=[]
            reg_ys=[]
        for x in range(max_step,size_x,max_step):
            for y in range(max_step,size_y,max_step):
                reg_xs.append(np.array([array[x-max_step][y-max_step],array[x-max_step][y],array[x][y-max_step]],dtype=np.float64))
                reg_ys.append(array[x][y])
                res=LinearRegression(fit_intercept=True).fit(reg_xs, reg_ys)
                coef=res.coef_ 
                ince=res.intercept_

 
   
        for x in range(0,size_x,max_step):
            for y in range(0,size_y,max_step):
                orig=array[x][y]
                if x and y and max_level>=min_coeff_level:
                    reg_block=np.array([array[x-max_step][y-max_step],array[x-max_step][y],array[x][y-max_step]],dtype=np.float64)
                    pred=np.dot(reg_block,coef)+ince

            
                
                else:
                    f_01=array[x-max_step][y] if x else 0
                    f_10=array[x][y-max_step] if y else 0
            
                    f_00=array[x-max_step][y-max_step] if x and y else 0
                
                    pred=f_01+f_10-f_00
                
        
                
                q,decomp=quantize(orig,pred,anchor_eb)
                qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y]=decomp
                predicted[x][y]=1
    else:
        anchor_eb=0
else:
    pass#raise some error
'''
#print(len(qs))

last_x=((size_x-1)//max_step)*max_step
last_y=((size_y-1)//max_step)*max_step   

#il_count=0
#ic_count=0
#im_count=0
#l_count=0
lorenzo_level=args.lorenzo_fallback_check
lorenzo_sample_ratio=args.fallback_sample_ratio
#currently no coeff and levelwise predictor selection.
if args.order=="block":
    for x_start in range(0,last_x,max_step):
        for y_start in range(0,last_y,max_step):
            x_end=size_x-1 if x_start==last_x-max_step else x_start+max_step 
            y_end=size_y-1 if y_start==last_y-max_step else y_start+max_step 
            #print(x_start,y_start)
            cur_qs,cur_lorenzo_qs,cur_us,cur_selected=\
            msc2d(array,x_start,x_end+1,y_start,y_end+1,error_bound,rate,maximum_rate,min_coeff_level,max_step,anchor_rate,\
                rate_list=rate_list,sz3_interp=args.sz_interp,multidim_level=args.multidim_level,lorenzo=args.lorenzo_fallback_check,\
                sample_rate=args.fallback_sample_ratio,min_sampled_points=10,x_preded=(x_start>0),y_preded=(y_start>0),random_access=False,fix_algo=args.fix)
        
     
        #pr=False

            for i in range(len(cur_qs)):
            
            #if (i==0 and len(cur_qs[0])!=3072):
                #pr=True
            #if pr:
                #print(len(cur_qs[i]))
                qs[i]+=cur_qs[i]

            us+=cur_us
        #if len(cur_lorenzo_qs)!=0:
        #    print("lor")
        #    print(len(cur_lorenzo_qs))
            lorenzo_qs+=cur_lorenzo_qs

        #if "lorenzo" in cur_selected[-1]:
            #print(x_start,y_start)
elif args.order=="level":
    num_blocks=((size_x-1)//max_step)*((size_y-1)//max_step)
    blocked_qs=[[ [] for i in range(max_level+1)] for i in range(num_blocks)]
    blocked_us=[[] for i in range(num_blocks)]
    max_level=int(math.log(args.max_step,2))

    for level in range(max_level-1,0,-1):
        idx=0
        for x_start in range(0,last_x,max_step):
            for y_start in range(0,last_y,max_step):
                
                x_end=size_x-1 if x_start==last_x-max_step else x_start+max_step 
                y_end=size_y-1 if y_start==last_y-max_step else y_start+max_step 
                #print(x_start,y_start)
                cur_qs,cur_lorenzo_qs,cur_us,cur_selected=\
                msc2d(array,x_start,x_end+1,y_start,y_end+1,error_bound,rate,maximum_rate,min_coeff_level,max_step,anchor_rate,\
                    rate_list=rate_list,sz3_interp=args.sz_interp,multidim_level=args.multidim_level,lorenzo=args.lorenzo_fallback_check,\
                    sample_rate=args.fallback_sample_ratio,min_sampled_points=10,x_preded=(x_start>0),y_preded=(y_start>0),random_access=False,fix_algo=args.fix,\
                    first_level=level,last_level=level,first_order="level")
                if "lorenzo" in cur_selected[-1]:
                    blocked_qs[idx]=[ [] for i in range(max_level+1)]
                    blocked_us[idx]=[]
                
                    
                blocked_qs[idx][level]=cur_qs[level]
                blocked_us[idx]+=cur_us
                if level==0:
                    lorenzo_qs+=cur_lorenzo_qs

                    



                idx+=1
    qs=[sum([blocked_qs[i][level] for i in range(num_blocks)],[]) for level in range(max_level+1)]
    us=sum(blocked_us,[])



else:
    print("Wrong order.")



 






quants=np.concatenate( (np.array(lorenzo_qs,dtype=np.int32),np.array(sum(qs,[]),dtype=np.int32) ) )
unpreds=np.array(us,dtype=np.float32)
array.tofile(args.output)
quants.tofile(args.quant)
unpreds.tofile(args.unpred)


'''
for x in range(size_x):
    for y in range(size_y):
        if array[x][y]==orig_array[x][y] and x%max_step!=0 and y%max_step!=0:
            print(x,y)
'''