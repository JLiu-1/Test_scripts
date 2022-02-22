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

def msc2d(array,error_bound,rate,maximum_rate,min_coeff_level,max_step,anchor_rate,rate_list=None,x_preded=False,y_preded=False,multidim=True,lorenzo=-1,\
sample_rate=0.05,min_sampled_points=10,random_access=False):#lorenzo:only check lorenzo fallback with level no larger than lorenzo level

    size_x,size_y=array.shape
    #array=np.fromfile(args.input,dtype=np.float32).reshape((size_x,size_y))
    if lorenzo>=0:
        orig_array=np.copy(array)
    if random_access and lorenzo>=0:
        lorenzo=0
    #error_bound=args.error*rng
    #max_step=args.max_step
    #rate=args.rate
    max_level=int(math.log(max_step,2))
    selected_algos=[]


    qs=[ [] for i in range(max_level+1)]

    us=[]
    edge_qs=[]
#min_coeff_level=args.min_coeff_level
#anchor=args.anchor
    if max_step>0:
    
    #anchor_rate=args.anchor_rate
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

 
            startx=max_step if x_preded else 0
            starty=max_step if y_preded else 0

            for x in range(startx,size_x,max_step):
                for y in range(starty,size_y,max_step):
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
                    qs[max_level].append(q)
                    if q==0:
                        us.append(decomp)
                    array[x][y]=decomp
        else:
            anchor_eb=0
    else:
        pass#raise error
#print(len(qs))

    last_x=((size_x-1)//max_step)*max_step
    last_y=((size_y-1)//max_step)*max_step   
    step=max_step//2
    level=max_level-1
    #maxlevel_q_start=len(qs[max_level])
    u_start=len(us)
    cumulated_loss=0.0
    while step>0:
        cur_qs=[]
        cur_us=[]
        if rate_list!=None:
            cur_eb=error_bound/rate_list[level]
        else:
            cur_eb=error_bound/min(maximum_rate,(rate**level))
        cur_array=np.copy(array[0:last_x+1:step,0:last_y+1:step])
        cur_size_x,cur_size_y=cur_array.shape
    #print(cur_size_x,cur_size_y)
        #print("Level %d started. Current step: %d. Current error_bound: %s." % (level,step,cur_eb))
        best_preds=None#need to copy
        best_absloss=None
        best_qs=[]#need to copy
        best_us=[]#need to copy
        xstart=2 if x_preded else 0
        ystart=2 if y_preded else 0
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
        

        for x in range(xstart,cur_size_x,2):
            for y in range(1,cur_size_y,2):
                if y==cur_size_y-1:
                    continue
                orig=cur_array[x][y]
                if level>=min_coeff_level:
                    pred= np.dot( np.array([cur_array[x][y-1],cur_array[x][y+1]]),coef )+ince 
                else:
                    pred=(cur_array[x][y-1]+cur_array[x][y+1])/2
                if (not random_access) or level!=0 or x!=cur_size_x-1 or last_x!=size_x-1:
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
                for y in range(ystart,cur_size_y,2):
                    reg_xs.append(np.array([cur_array[x-1][y],cur_array[x+1][y]],dtype=np.float64))
                    reg_ys.append(cur_array[x][y])
                    res=LinearRegression(fit_intercept=True).fit(reg_xs, reg_ys)
                    coef=res.coef_ 
                    ince=res.intercept_
        for x in range(1,cur_size_x,2):
            for y in range(ystart,cur_size_y,2):
                if x==cur_size_x-1:
                    continue
                orig=cur_array[x][y]
                if level>=min_coeff_level:
                    pred= np.dot( np.array([cur_array[x-1][y],cur_array[x+1][y]]),coef )+ince 
                else:
                    pred=(cur_array[x-1][y]+cur_array[x+1][y])/2
                if (not random_access) or level!=0 or y!=cur_size_y-1 or last_y!=size_y-1:
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
        #cubic=True
        #if cubic:
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
        for x in range(xstart,cur_size_x,2):
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
                if (not random_access) or level!=0 or x!=cur_size_x-1 or last_x!=size_x-1:
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
            for y in range(ystart,cur_size_y,2):
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
                if (not random_access) or level!=0 or y!=cur_size_y-1 or last_y!=size_y-1:
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
        if multidim:
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
                if x==0 and xstart!=0:
                    continue
                for y in range(1-(x%2),cur_size_y,2):
                    if y==0 and ystart!=0:
                        continue
                
                    orig=cur_array[x][y]
                    if x and y and x!=cur_size_x-1 and y!=cur_size_y-1:
                        if level>=min_coeff_level:
                            pred=np.dot(md_coef,np.array([cur_array[x][y-1],cur_array[x][y+1],cur_array[x-1][y],cur_array[x+1][y]]))+md_ince
                    
                        else:

                            pred=(cur_array[x][y-1]+cur_array[x][y+1]+cur_array[x-1][y]+cur_array[x+1][y])/4
                    elif x==0 or x==cur_size_x-1:
                        pred=(cur_array[x][y-1]+cur_array[x][y+1])/2
                    else:
                        pred=(cur_array[x-1][y]+cur_array[x+1][y])/2
                    if (not random_access) or level!=0 or (x!=cur_size_x-1 or last_x!=size_x-1) or (y!=cur_size_y-1 or last_y!=size_y-1):
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
    
        #Lorenzo fallback
        if level<=lorenzo:
            absloss=0
        #cur_qs=[]
        #cur_us=[]
        #cur_array=np.copy(array[0:last_x+1:step,0:last_y+1:step])#reset cur_array
            xstart=1 if x_preded else 0
            ystart=1 if y_preded else 0
            cur_orig_array=orig_array[0:last_x+1:step,0:last_y+1:step]
            x_end_offset=1 if (random_access and last_x==size_x-1 and not x_edge and level==0) else 0
            y_end_offset=1 if (random_access and last_y==size_y-1 and not y_edge and level==0) else 0
            total_points=[(x,y) for x in range(cur_orig_array.shape[0]-1) for y in range(cur_orig_array.shape[1]-1) if (max_step<=0 or ((x*step)%max_step!=0 and (y*step)%max_step!=0))]
            if len(total_points)<min_sampled_points:
                num_sumples=len(total_points)
                sampled_points=total_points
            else:
                num_sumples=max(min_sampled_points,int(len(total_points)*sample_rate) )
                sampled_points=random.sample(total_points,num_sumples)
            for x,y in sampled_points:
                orig=cur_orig_array[x][y]
                f_01=cur_orig_array[x-1][y] if x else 0
                if x and max_step>0 and ((x-1)*step)%max_step==0 and (y*step)%max_step==0:
                    f_01+=anchor_eb*(2*np.random.rand()-1)
                elif x:
                    f_01+=cur_eb*(2*np.random.rand()-1)

                f_10=cur_orig_array[x][y-1] if y else 0
                if y and max_step>0 and (x*step)%max_step==0 and ((y-1)*step)%max_step==0:
                    f_10+=anchor_eb*(2*np.random.rand()-1)
                elif y:
                    f_10+=cur_eb*(2*np.random.rand()-1)
            
                f_00=cur_orig_array[x-1][y-1] if x and y else 0
                if x and y and max_step>0 and ((x-1)*step)%max_step==0 and ((y-1)*step)%max_step==0:
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
           
            #qs[max_level]=qs[:maxlevel_q_start]
                for i in range(max_level-1,level,-1):
                    qs[i]=[]
                us=us[:u_start]
                for x in range(xstart,cur_size_x-x_end_offset):
                    for y in range(ystart,cur_size_y-y_end_offset):
                    
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
        array[0:last_x+1:step,0:last_y+1:step]=best_preds
        if selected_algo!="lorenzo_fallback":
            cumulated_loss+=best_absloss
        
        else:
            cumulated_loss=best_absloss
        
        #print(np.max(np.abs(array[0:last_x+1:step,0:last_y+1:step]-best_preds)))
    
        #if args.lorenzo_fallback_check:
        #    print(np.max(np.abs(orig_array-array))/rng)
        qs[level]+=best_qs
        us+=best_us
        selected_algos.append(selected_algo)
        #print(len(qs))
        #print ("Level %d finished. Selected algorithm: %s. Mean prediction abs loss: %f." % (level,selected_algo,mean_l1_loss))
        step=step//2
        level-=1
        #print(sum([len(_) for _ in qs] ))
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
                edge_qs.append(q)
                if q==0:
                    us.append(decomp)
                array[x][y]=decomp
    offset_x1=1 if x_preded else 0
    offset_y1=1 if y_preded else 0
    offset_x2=1 if random_access else 0
    offset_y2=1 if random_access else 0
    lorenzo_2d(array,offset_x1,last_x+1,last_y+1,size_y-offset_y2)
    lorenzo_2d(array,last_x+1,size_x-offset_x2,offset_y1,size_y-offset_y2)
    return array,qs,edge_qs,us,selected_algos


    
if __name__=="__main__":
 



    parser = argparse.ArgumentParser()

    parser.add_argument('--error','-e',type=float,default=1e-3)
    parser.add_argument('--input','-i',type=str)
    parser.add_argument('--output','-o',type=str)
    parser.add_argument('--quant','-q',type=str,default="ml2_q2.dat")
    parser.add_argument('--unpred','-u',type=str,default="ml2_u2.dat")
    parser.add_argument('--max_step','-s',type=int,default=-1)
    parser.add_argument('--min_coeff_level','-cl',type=int,default=99)
    parser.add_argument('--rate','-r',type=float,default=1.0)
    parser.add_argument('--rlist',type=float,default=0.0,nargs="+")
    parser.add_argument('--maximum_rate','-m',type=float,default=10.0)
    parser.add_argument('--cubic','-c',type=int,default=1)
    parser.add_argument('--multidim','-d',type=int,default=1)
    parser.add_argument('--lorenzo_fallback_check','-l',type=int,default=-1)
    parser.add_argument('--fallback_sample_ratio','-f',type=float,default=0.01)
#parser.add_argument('--level_rate','-lr',type=float,default=1.0)
    parser.add_argument('--anchor_rate','-a',type=float,default=0.0)

    parser.add_argument('--size_x','-x',type=int,default=1800)
    parser.add_argument('--size_y','-y',type=int,default=3600)
#parser.add_argument('--level','-l',type=int,default=2)
#parser.add_argument('--noise','-n',type=bool,default=False)
#parser.add_argument('--intercept','-t',type=bool,default=False)
    args = parser.parse_args()
    array=np.fromfile(args.input,dtype=np.float32).reshape((args.size_x,args.size_y))
    orig_array=np.copy(array)
    error_bound=args.error*(np.max(array)-np.min(array))
    max_level=int(math.log(args.max_step,2))
    if args.rlist!=0:
        rate_list=args.rlist
        if isinstance(rate_list,int):
            rate_list=[rate_list]

        while len(rate_list)<max_level:
            rate_list.insert(0,rate_list[0])
    else:
        rate_list=None
    array,qs,edge_qs,us,_=msc2d(array,error_bound,args.rate,args.maximum_rate,args.min_coeff_level,args.max_step,args.anchor_rate,rate_list=rate_list,x_preded=False,y_preded=False,multidim=args.multidim,\
        lorenzo=args.lorenzo_fallback_check,sample_rate=args.fallback_sample_ratio,min_sampled_points=100,random_access=False)

    quants=np.concatenate( (np.array(edge_qs,dtype=np.int32),np.array(sum(qs,[]),dtype=np.int32) ) )
    unpreds=np.array(us,dtype=np.float32)
    array.tofile(args.output)
    quants.tofile(args.quant)
    unpreds.tofile(args.unpred)
    '''
    for x in range(args.size_x):
        for y in range(args.size_y):
            if array[x][y]==orig_array[x][y] and x%args.max_step!=0 and y%args.max_step!=0:
                print(x,y)
    '''