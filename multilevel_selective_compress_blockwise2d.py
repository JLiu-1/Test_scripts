import numpy as np 

import os
import argparse
#import torch
#import torch.nn as nn
from sklearn.linear_model import LinearRegression
import math
import random
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
parser.add_argument('--max_step','-s',type=int,default=64)
#parser.add_argument('--min_coeff_level','-cl',type=int,default=99)
parser.add_argument('--rate','-r',type=float,default=1.0)
parser.add_argument('--maximum_rate','-m',type=float,default=10.0)
#parser.add_argument('--cubic','-c',type=int,default=1)
parser.add_argument('--multidim','-d',type=int,default=1)
#parser.add_argument('--lorenzo_fallback_check','-l',type=int,default=0)
#parser.add_argument('--fallback_sample_ratio','-f',type=float,default=0.01)
#parser.add_argument('--level_rate','-lr',type=float,default=1.0)
parser.add_argument('--anchor_rate','-a',type=float,default=0.0)

parser.add_argument('--size_x','-x',type=int,default=1800)
parser.add_argument('--size_y','-y',type=int,default=3600)
#parser.add_argument('--level','-l',type=int,default=2)
#parser.add_argument('--noise','-n',type=bool,default=False)
#parser.add_argument('--intercept','-t',type=bool,default=False)
args = parser.parse_args()

size_x=args.size_x
size_y=args.size_y
array=np.fromfile(args.input,dtype=np.float32).reshape((size_x,size_y))
#if args.lorenzo_fallback_check:
    #orig_array=np.copy(array)
predicted=np.zeros((size_x,size_y),dtype=np.int16)
rng=(np.max(array)-np.min(array))
error_bound=args.error*rng
max_step=args.max_step
rate=args.rate
maximum_rate=args.maximum_rate


max_level=int(math.log(max_step,2))

qs=[[] for i in range(max_level+1)]

us=[]
lorenzo_qs=[]
min_coeff_level=args.min_coeff_level
#anchor=args.anchor
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
#print(len(qs))

last_x=((size_x-1)//max_step)*max_step
last_y=((size_y-1)//max_step)*max_step   

il_count=0
ic_count=0
im_count=0
l_count=0

#currently no coeff and levelwise predictor selection.
for x_start in range(0,last_x,max_step):
    for y_start in range(0,last_y,max_step):
        x_end=x_start+max_step
        y_end=y_start+max_step
        best_absloss=None
        selected_algo="none"
        cur_absloss=0
        #pred_count=0
        #interp linear
        cur_array=np.copy(array[x_start:x_end+1,y_start:y_end+1])
        step=max_step//2
        cur_level=max_level-1
        while step>0:

            cur_rate=min(maximum_rate,rate**cur_level)
            cur_eb=error_bound/cur_rate
            doublestep=2*step
            #triplestep=3*step
            startx=doublestep if predicted[x_start][y_start+1] else 0
            starty=doublestep if predicted[x_start+1][y_start] else 0
            for x in range(startx,x_end+1,doublestep):


                for y in range(step,y_end+1,doublestep):
                    orig=cur_array[x][y]
          
                    pred=(cur_array[x][y-1]+cur_array[x][y+1])/2
                    if x!=x_end and y!=y_end:
                        cur_absloss+=abs(orig-pred)
                        #pred_count+=1
            
                    q,decomp=quantize(orig,pred,cur_eb)
                    cur_array[x][y]=decomp 
            for x in range(step,x_end+1,doublestep):


                for y in range(ystart,y_end+1,doublestep):
                    orig=cur_array[x][y]
          
                    pred=(cur_array[x-1][y]+cur_array[x+1][y])/2
                    if x!=x_end and y!=y_end:
                        cur_absloss+=abs(orig-pred)
                        #pred_count+=1
            
                    q,decomp=quantize(orig,pred,cur_eb)
                    cur_array[x][y]=decomp 
            for x in range(xstart,x_end+1,doublestep):


                for y in range(ystart,y_end+1,doublestep):
                    orig=cur_array[x][y]
          
                    pred=(cur_array[x][y-1]+cur_array[x][y+1]+cur_array[x-1][y]+cur_array[x+1][y])/4
                    if x!=x_end and y!=y_end:
                        cur_absloss+=abs(orig-pred)
                        #pred_count+=1
            
                    q,decomp=quantize(orig,pred,cur_eb)
                    cur_array[x][y]=decomp 

            step=step//2
            level-=1  
        selected_algo="interp_linear"
        best_absloss=cur_absloss

        #interp_cubic
        cur_array=np.copy(array[x_start:x_end+1,y_start:y_end+1])
        step=max_step//2
        cur_level=max_level-1
        while step>0:

            cur_rate=min(maximum_rate,rate**cur_level)
            cur_eb=error_bound/cur_rate
            doublestep=2*step
            triplestep=3*step
            startx=doublestep if predicted[x_start][y_start+1] else 0
            starty=doublestep if predicted[x_start+1][y_start] else 0
            for x in range(startx,x_end+1,doublestep):


                for y in range(step,y_end+1,doublestep):
                    orig=cur_array[x][y]
          
                    pred=(cur_array[x][y-1]+cur_array[x][y+1])/2
                    if x!=x_end and y!=y_end:
                        cur_absloss+=abs(orig-pred)
                        #pred_count+=1
            
                    q,decomp=quantize(orig,pred,cur_eb)
                    cur_array[x][y]=decomp 
            for x in range(step,x_end+1,doublestep):


                for y in range(ystart,y_end+1,doublestep):
                    orig=cur_array[x][y]
          
                    pred=(cur_array[x-1][y]+cur_array[x+1][y])/2
                    if x!=x_end and y!=y_end:
                        cur_absloss+=abs(orig-pred)
                        #pred_count+=1
            
                    q,decomp=quantize(orig,pred,cur_eb)
                    cur_array[x][y]=decomp 
            for x in range(xstart,x_end+1,doublestep):


                for y in range(ystart,y_end+1,doublestep):
                    orig=cur_array[x][y]
          
                    pred=(cur_array[x][y-1]+cur_array[x][y+1]+cur_array[x-1][y]+cur_array[x+1][y])/4
                    if x!=x_end and y!=y_end:
                        cur_absloss+=abs(orig-pred)
                        #pred_count+=1
            
                    q,decomp=quantize(orig,pred,cur_eb)
                    cur_array[x][y]=decomp 

            step=step//2
            level-=1  


           








step=max_step//2
level=max_level-1
q_start=len(qs)
u_start=len(us)
cumulated_loss=0.0
while step>0:
    cur_qs=[]
    cur_us=[]
    cur_eb=error_bound/min(args.maximum_rate,(rate**level))
    cur_array=np.copy(array[0:last_x+1:step,0:last_y+1:step])
    cur_size_x,cur_size_y=cur_array.shape
    #print(cur_size_x,cur_size_y)
    print("Level %d started. Current step: %d. Current error_bound: %s." % (level,step,cur_eb))
    best_preds=None#need to copy
    best_absloss=None
    best_qs=[]#need to copy
    best_us=[]#need to copy

    #linear interp
    absloss=0
    selected_algo="none"
    
    if level>=min_coeff_level:
        reg_xs=[]
        reg_ys=[]
        for x in range(0,cur_size_x,2):
            for y in range(1,cur_size_y,2):
                reg_xs.append(np.array([cur_array[x][y-1],cur_array[x][y+1]],dtype=np.float64))
                reg_ys.append(cur_array[x][y])
                res=LinearRegression(fit_intercept=True).fit(reg_xs, reg_ys)
                coef=res.coef_ 
                ince=res.intercept_
 

    for x in range(0,cur_size_x,2):
        for y in range(1,cur_size_y,2):
            if y==cur_size_y-1:
                continue
            orig=cur_array[x][y]
            if level>=min_coeff_level:
                pred= np.dot( np.array([cur_array[x][y-1],cur_array[x][y+1]]),coef )+ince 
            else:
                pred=(cur_array[x][y-1]+cur_array[x][y+1])/2
            absloss+=abs(orig-pred)
            q,decomp=quantize(orig,pred,cur_eb)
            cur_qs.append(q)
            

            if q==0:
                cur_us.append(decomp)
                #absloss+=abs(decomp)
            cur_array[x][y]=decomp     
    if level>=min_coeff_level:
        reg_xs=[]
        reg_ys=[]
        for x in range(1,cur_size_x,2):
            for y in range(0,cur_size_y,2):
                reg_xs.append(np.array([cur_array[x-1][y],cur_array[x+1][y]],dtype=np.float64))
                reg_ys.append(cur_array[x][y])
                res=LinearRegression(fit_intercept=True).fit(reg_xs, reg_ys)
                coef=res.coef_ 
                ince=res.intercept_
    for x in range(1,cur_size_x,2):
        for y in range(0,cur_size_y,2):
            if x==cur_size_x-1:
                continue
            orig=cur_array[x][y]
            if level>=min_coeff_level:
                pred= np.dot( np.array([cur_array[x-1][y],cur_array[x+1][y]]),coef )+ince 
            else:
                pred=(cur_array[x-1][y]+cur_array[x+1][y])/2
            absloss+=abs(orig-pred)
            q,decomp=quantize(orig,pred,cur_eb)
           
            cur_qs.append(q)
            if q==0:
                cur_us.append(decomp)
                #absloss+=abs(decomp)
            cur_array[x][y]=decomp
    if level>=min_coeff_level:
        md_reg_xs=[]
        md_reg_ys=[]
        for x in range(1,cur_size_x,2):
            for y in range(1,cur_size_y,2):
                md_reg_xs.append(np.array([cur_array[x-1][y],cur_array[x+1][y],cur_array[x][y-1],cur_array[x][y+1]],dtype=np.float64))
                md_reg_ys.append(cur_array[x][y])
                md_res=LinearRegression(fit_intercept=True).fit(md_reg_xs, md_reg_ys)
                md_coef=md_res.coef_ 
                md_ince=md_res.intercept_

    
    for x in range(1,cur_size_x,2):
        for y in range(1,cur_size_y,2):
            if x==cur_size_x-1 or y==cur_size_y-1:
                continue
            orig=cur_array[x][y]
            if level>=min_coeff_level:
                pred=np.dot(np.array([cur_array[x-1][y],cur_array[x+1][y],cur_array[x][y-1],cur_array[x][y+1]]),md_coef)+md_ince
            else:
                pred=(cur_array[x-1][y]+cur_array[x+1][y]+cur_array[x][y-1]+cur_array[x][y+1])/4
            absloss+=abs(orig-pred)
            q,decomp=quantize(orig,pred,cur_eb)
            
            cur_qs.append(q)
            if q==0:
                cur_us.append(decomp)
                #absloss+=abs(decomp)
            cur_array[x][y]=decomp

    best_preds=np.copy(cur_array)
    best_absloss=absloss
    best_qs=cur_qs.copy()
    best_us=cur_us.copy()
    selected_algo="interp_linear"
    #print(len(cur_qs))
    #cubic interp
    if args.cubic:
        #print("cubic")
        absloss=0
        cur_qs=[]
        cur_us=[]
        cur_array=np.copy(array[0:last_x+1:step,0:last_y+1:step])#reset cur_array
        if level>=min_coeff_level:
            reg_xs=[]
            reg_ys=[]
            for x in range(0,cur_size_x,2):
                for y in range(3,cur_size_y,2):
                    if y+3>=cur_size_y:
                        continue
                    reg_xs.append(np.array([cur_array[x][y-3],cur_array[x][y-1],cur_array[x][y+1],cur_array[x][y+3]],dtype=np.float64))
                    reg_ys.append(cur_array[x][y])
                    res=LinearRegression(fit_intercept=True).fit(reg_xs, reg_ys)
                    coef=res.coef_ 
                    ince=res.intercept_
        for x in range(0,cur_size_x,2):
            for y in range(1,cur_size_y,2):
                if y==cur_size_y-1:
                    continue
                orig=cur_array[x][y]
                if y>=3 and y+3<cur_size_y:
                    if level>=min_coeff_level:
                        pred=np.dot(coef,np.array([cur_array[x][y-3],cur_array[x][y-1],cur_array[x][y+1],cur_array[x][y+3]]) )+ince
                    else:
                        pred=(-cur_array[x][y-3]+9*cur_array[x][y-1]+9*cur_array[x][y+1]-cur_array[x][y+3])/16
                else:
                    pred=(cur_array[x][y-1]+cur_array[x][y+1])/2
                absloss+=abs(orig-pred)
                q,decomp=quantize(orig,pred,cur_eb)
                cur_qs.append(q)
                
                if q==0:
                    cur_us.append(decomp)
                    #absloss+=abs(decomp)
                cur_array[x][y]=decomp     
        if level>=min_coeff_level:
            reg_xs=[]
            reg_ys=[]
            for x in range(3,cur_size_x,2):
                for y in range(0,cur_size_y,2):
                    if x+3>=cur_size_x:
                        continue
                    reg_xs.append(np.array([cur_array[x-3][y],cur_array[x-1][y],cur_array[x+1][y],cur_array[x+3][y]],dtype=np.float64))
                    reg_ys.append(cur_array[x][y])
                    res=LinearRegression(fit_intercept=True).fit(reg_xs, reg_ys)
                    coef=res.coef_ 
                    ince=res.intercept_
        for x in range(1,cur_size_x,2):
            for y in range(0,cur_size_y,2):
                if x==cur_size_x-1:
                    continue
                orig=cur_array[x][y]
                if x>=3 and x+3<cur_size_x:
                    if level>=min_coeff_level:
                        pred=np.dot(coef,np.array([cur_array[x-3][y],cur_array[x-1][y],cur_array[x+1][y],cur_array[x+3][y]]) )+ince
                    else:
                        pred=(-cur_array[x-3][y]+9*cur_array[x-1][y]+9*cur_array[x+1][y]-cur_array[x+3][y])/16
                else:
                    pred=(cur_array[x-1][y]+cur_array[x+1][y])/2
                absloss+=abs(orig-pred)
                q,decomp=quantize(orig,pred,cur_eb)
                
                cur_qs.append(q)
                if q==0:
                    cur_us.append(decomp)
                    #absloss+=abs(decomp)
                cur_array[x][y]=decomp
        if level>=min_coeff_level:
            md_reg_xs=[]
            md_reg_ys=[]
            for x in range(1,cur_size_x,2):
                for y in range(1,cur_size_y,2):
                    md_reg_xs.append(np.array([cur_array[x-1][y],cur_array[x+1][y],cur_array[x][y-1],cur_array[x][y+1]],dtype=np.float64))
                    md_reg_ys.append(cur_array[x][y])
                    md_res=LinearRegression(fit_intercept=True).fit(md_reg_xs, md_reg_ys)
                    md_coef=md_res.coef_ 
                    md_ince=md_res.intercept_

        for x in range(1,cur_size_x,2):
            for y in range(1,cur_size_y,2):
                if x==cur_size_x-1 or y==cur_size_y-1:
                    continue
                orig=cur_array[x][y]
                if level>=min_coeff_level:
                    pred=np.dot(np.array([cur_array[x-1][y],cur_array[x+1][y],cur_array[x][y-1],cur_array[x][y+1]]),md_coef)+md_ince
                else:
                    pred=(cur_array[x-1][y]+cur_array[x+1][y]+cur_array[x][y-1]+cur_array[x][y+1])/4
                absloss+=abs(orig-pred)
                q,decomp=quantize(orig,pred,cur_eb)
                
                cur_qs.append(q)
                if q==0:
                    cur_us.append(decomp)
                    #absloss+=abs(decomp)
                cur_array[x][y]=decomp
        if absloss<best_absloss:
            selected_algo="interp_cubic"
            best_preds=np.copy(cur_array)
            best_absloss=absloss
            best_qs=cur_qs.copy()
            best_us=cur_us.copy()
    if args.multidim:
        absloss=0
        cur_qs=[]
        cur_us=[]
        cur_array=np.copy(array[0:last_x+1:step,0:last_y+1:step])#reset cur_array
        if level>=min_coeff_level:
            md_reg_xs=[]
            md_reg_ys=[]
            for x in range(1,cur_size_x,2):
                for y in range(1,cur_size_y,2):
                    md_reg_xs.append(np.array([cur_array[x-1][y-1],cur_array[x-1][y+1],cur_array[x+1][y-1],cur_array[x+1][y+1]],dtype=np.float64))
                    md_reg_ys.append(cur_array[x][y])
                    md_res=LinearRegression(fit_intercept=True).fit(md_reg_xs, md_reg_ys)
                    md_coef=md_res.coef_ 
                    md_ince=md_res.intercept_
        for x in range(1,cur_size_x,2):
            for y in range(1,cur_size_y,2):
                if x==cur_size_x-1 or y==cur_size_y-1:
                    continue
                orig=cur_array[x][y]
                if level>=min_coeff_level:
                    pred=np.dot(np.array([cur_array[x-1][y-1],cur_array[x-1][y+1],cur_array[x+1][y-1],cur_array[x+1][y+1]]),md_coef)+md_ince
                else:
                    pred=(cur_array[x-1][y-1]+cur_array[x-1][y+1]+cur_array[x+1][y-1]+cur_array[x+1][y+1])/4
                absloss+=abs(orig-pred)
                q,decomp=quantize(orig,pred,cur_eb)
            
                cur_qs.append(q)
                if q==0:
                    cur_us.append(decomp)
                    #absloss+=abs(decomp)
                cur_array[x][y]=decomp
        if level>=min_coeff_level:
            md_reg_xs=[]
            md_reg_ys=[]
            for x in range(0,cur_size_x):
                for y in range(1-(x%2),cur_size_y,2):
                    if x==cur_size_x-1 or y==cur_size_y-1:
                        continue
                    md_reg_xs.append(np.array([cur_array[x][y-1],cur_array[x][y+1],cur_array[x-1][y],cur_array[x+1][y]],dtype=np.float64))
                    md_reg_ys.append(cur_array[x][y])
                    md_res=LinearRegression(fit_intercept=True).fit(md_reg_xs, md_reg_ys)
                    md_coef=md_res.coef_ 
                    md_ince=md_res.intercept_

        for x in range(0,cur_size_x):
            for y in range(1-(x%2),cur_size_y,2):
                if x==cur_size_x-1 or y==cur_size_y-1:
                    continue
                orig=cur_array[x][y]
                if x and y:
                    if level>=min_coeff_level:
                        pred=np.dot(md_coef,np.array([cur_array[x][y-1],cur_array[x][y+1],cur_array[x-1][y],cur_array[x+1][y]]))+md_ince
                    
                    else:

                        pred=(cur_array[x][y-1]+cur_array[x][y+1]+cur_array[x-1][y]+cur_array[x+1][y])/4
                elif x==0:
                    pred=(cur_array[x][y-1]+cur_array[x][y+1])/2
                else:
                    pred=(cur_array[x-1][y]+cur_array[x+1][y])/2
                absloss+=abs(orig-pred)
                q,decomp=quantize(orig,pred,cur_eb)
                cur_qs.append(q)
            

                if q==0:
                    cur_us.append(decomp)
                #absloss+=abs(decomp)
                cur_array[x][y]=decomp
        if absloss<best_absloss:
            selected_algo="interp_multidim"
            best_preds=np.copy(cur_array)
            best_absloss=absloss
            best_qs=cur_qs.copy()
            best_us=cur_us.copy()
    #lorenzo
    '''
    cur_array=np.copy(array[0:last_x+1:step,0:last_y+1:step])#reset cur_array
    absloss=0
    cur_qs=[]
    cur_us=[]
    
    if max_level>=min_coeff_level:
        reg_xs=[]
        reg_ys=[]
        for x in range(cur_size_x):
            for y in range(1-(x%2),cur_size_y,2-(x%2)):
                if not (x and y):
                    continue
                reg_xs.append(np.array([array[x-1][y-1],array[x-1][y],array[x][y-1]],dtype=np.float64))
                reg_ys.append(array[x][y])
                res=LinearRegression(fit_intercept=True).fit(reg_xs, reg_ys)
                coef=res.coef_ 
                ince=res.intercept_
    
    for x in range(cur_size_x):
        for y in range(1-(x%2),cur_size_y,2-(x%2)):
            orig=cur_array[x][y]
            if 0:#if x and y and max_level>=min_coeff_level:
                pred=np.dot(coef,np.array([array[x-1][y-1],array[x-1][y],array[x][y-1]]))+ince
            else:

                f_01=cur_array[x-1][y] if x else 0
                
                f_10=cur_array[x][y-1] if y else 0
            
                f_00=cur_array[x-1][y-1] if x and y else 0
                
                pred=f_01+f_10-f_00
                
        
            absloss+=abs(orig-pred)
            q,decomp=quantize(orig,pred,cur_eb)
            cur_qs.append(q)
            
            if q==0:
                cur_us.append(decomp)
                #absloss+=abs(decomp)
            cur_array[x][y]=decomp
    #print(np.max(np.abs(array[0:last_x+1:step,0:last_y+1:step]-cur_array)))
    if absloss<best_absloss:
        best_preds=np.copy(cur_array)
        best_absloss=absloss
        best_qs=cur_qs.copy()
        best_us=cur_us.copy()
        selected_algo="lorenzo"
    '''
    #Lorenzo fallback
    if args.lorenzo_fallback_check:
        absloss=0
        #cur_qs=[]
        #cur_us=[]
        #cur_array=np.copy(array[0:last_x+1:step,0:last_y+1:step])#reset cur_array
        cur_orig_array=orig_array[0:last_x+1:step,0:last_y+1:step]
        total_points=[(x,y) for x in range(cur_orig_array.shape[0]) for y in range(cur_orig_array.shape[1]) if (max_step<=0 or ((x*step)%max_step!=0 and (y*step)%max_step!=0))]
        if len(total_points)<=100:
            num_sumples=total_points
        else:
            num_sumples=max(100,int(len(total_points)*args.fallback_sample_ratio) )
        sampled_points=random.sample(total_points,num_sumples)
        for x,y in sampled_points:
            orig=cur_orig_array[x][y]
            f_01=cur_orig_array[x-1][y] if x else 0
            if x and args.max_step>0 and ((x-1)*step)%max_step==0 and (y*step)%max_step==0:
                f_01+=anchor_eb*(2*np.random.rand()-1)
            elif x:
                f_01+=cur_eb*(2*np.random.rand()-1)

            f_10=cur_orig_array[x][y-1] if y else 0
            if y and args.max_step>0 and (x*step)%max_step==0 and ((y-1)*step)%max_step==0:
                f_10+=anchor_eb*(2*np.random.rand()-1)
            elif y:
                f_10+=cur_eb*(2*np.random.rand()-1)
            
            f_00=cur_orig_array[x-1][y-1] if x and y else 0
            if x and y and args.max_step>0 and ((x-1)*step)%max_step==0 and ((y-1)*step)%max_step==0:
                f_00+=anchor_eb*(2*np.random.rand()-1)
            elif x and y:
                f_00+=cur_eb*(2*np.random.rand()-1)
                
            pred=f_01+f_10-f_00

            absloss+=abs(orig-pred)
        #print(absloss*len(total_points)/len(sampled_points))
        #print(best_absloss)
        #print(cumulated_loss)
        if absloss*len(total_points)/len(sampled_points)<best_absloss+cumulated_loss:
            selected_algo="lorenzo_fallback"
            best_absloss=0
            best_preds=array[0:last_x+1:step,0:last_y+1:step]
            best_qs=[]
            best_us=[]
           
            qs=qs[:q_start]
            us=us[:u_start]
            for x in range(cur_size_x):
                for y in range(cur_size_y):
                    
                    if max_step>0 and (x*step)%max_step==0 and (y*step)%max_step==0:
                        #print(x,y)
                        continue
                    orig=best_preds[x][y]
                    f_01=best_preds[x-1][y] if x else 0
                
                    f_10=best_preds[x][y-1] if y else 0
            
                    f_00=best_preds[x-1][y-1] if x and y else 0
                
                    pred=f_01+f_10-f_00
                
        
                    best_absloss+=abs(orig-pred)
                    q,decomp=quantize(orig,pred,cur_eb)
                    best_qs.append(q)
                    if q==0:
                        best_us.append(decomp)
                #absloss+=abs(decomp)
                    best_preds[x][y]=decomp
            

        #print(len(best_qs))




    mean_l1_loss=best_absloss/len(best_qs)
    if selected_algo!="lorenzo_fallback":
        cumulated_loss+=best_absloss
    else:
        cumulated_loss=best_absloss
    #print(np.max(np.abs(array[0:last_x+1:step,0:last_y+1:step]-best_preds)))
    array[0:last_x+1:step,0:last_y+1:step]=best_preds
    #if args.lorenzo_fallback_check:
    #    print(np.max(np.abs(orig_array-array))/rng)
    qs+=best_qs
    us+=best_us
    #print(len(qs))
    print ("Level %d finished. Selected algorithm: %s. Mean prediction abs loss: %f." % (level,selected_algo,mean_l1_loss))
    step=step//2
    level-=1
    #print(best_absloss)
    #print(cumulated_loss)



def lorenzo_2d(array,x_start,x_end,y_start,y_end):
    for x in range(x_start,x_end):
        for y in range(y_start,y_end):

            orig=array[x][y]
        
            f_01=array[x-1][y] if x else 0
            f_10=array[x][y-1] if y else 0
            
            f_00=array[x-1][y-1] if x and y else 0
                
            pred=f_01+f_10-f_00
                
        
                
            q,decomp=quantize(orig,pred,error_bound)
            lorenzo_qs.append(q)
            if q==0:
                us.append(decomp)
            array[x][y]=decomp
lorenzo_2d(array,0,last_x+1,last_y+1,size_y)
lorenzo_2d(array,last_x+1,size_x,0,size_y)




 






quants=np.concatenate( (np.array(lorenzo_qs,dtype=np.int32),np.array(qs,dtype=np.int32) ) )
unpreds=np.array(us,dtype=np.float32)
array.tofile(args.output)
quants.tofile(args.quant)
unpreds.tofile(args.unpred)