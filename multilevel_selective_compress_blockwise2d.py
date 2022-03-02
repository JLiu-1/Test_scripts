import numpy as np 

import os
import argparse
#import torch
#import torch.nn as nn
from sklearn.linear_model import LinearRegression
import math
import random
from multilevel_selective_compress_2d_api import msc2d
from utils import *

if __name__=="__main__":

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
    parser.add_argument('--multidim_level','-d',type=int,default=-1)
    parser.add_argument('--lorenzo_fallback_check','-l',type=int,default=0)
    parser.add_argument('--fallback_sample_ratio','-p',type=float,default=0.05)
    parser.add_argument('--anchor_rate','-a',type=float,default=0.0)

    parser.add_argument('--size_x','-x',type=int,default=1800)
    parser.add_argument('--size_y','-y',type=int,default=3600)
    parser.add_argument('--sz_interp','-n',type=int,default=0)
    parser.add_argument('--fix_algo','-f',type=str,default="none")
    parser.add_argument('--autotuning','-t',type=float,default=0.0)
    args = parser.parse_args()

    size_x=args.size_x
    size_y=args.size_y
    array=np.fromfile(args.input,dtype=np.float32).reshape((size_x,size_y))
    orig_array=np.copy(array)
    #if args.lorenzo_fallback_check:
        #orig_array=np.copy(array)
    #predicted=np.zeros((size_x,size_y),dtype=np.int16)
    rng=(np.max(array)-np.min(array))
    error_bound=args.error*rng
    max_step=args.max_step
    rate=args.rate
    maximum_rate=args.maximum_rate
    anchor_rate=args.anchor_rate
    max_level=int(math.log(max_step,2))
    rate_list=args.rlist
    if args.autotuning!=0 and args.rlist!=-1:
        #pid=os.getpid()
        alpha_list=[1,1.25,1.5,1.75,2]
        beta_list=[1.5,2,3,4]
        rate_list=None
        block_num_x=(args.size_x-1)//args.max_step
        block_num_y=(args.size_y-1)//args.max_step
        steplength=int(math.sqrt(args.autotuning))
        bestalpha=1
        bestbeta=1
        #bestpdb=0
        bestb=9999
        #bestb_r=9999
        bestp=0
        #bestp_r=0
        pid=os.getpid()
        tq_name="%s_tq.dat"%pid
        tu_name="%s_tu.dat"%pid
        max_step=args.max_step
        max_level=int(math.log(max_step,2))
        for k,alpha in enumerate(alpha_list):
            for beta in beta_list:
                if alpha>beta:
                    continue
                #maybe some pruning
                test_qs=[[] for i in range(max_level+1)]
                test_us=[]
                square_error=0
                #zero_square_error=0
                element_counts=0
                themax=-9999999999999
                themin=99999999999999
                #themean=0
                #print(themean)
                for i in range(0,block_num_x,steplength):
                    for j in range(0,block_num_y,steplength):
                          
                        x_start=max_step*i
                        y_start=max_step*j
                        x_end=x_start+max_step+1
                        y_end=y_start+max_step+1
                            #print(x_start)
                            #print(y_start)
                        cur_array=np.copy(array[x_start:x_end,y_start:y_end])
                            
                            #left question: The predictor selection is separated on each block, which does not follow the real compression
                            #What about fix the prediction on SZ3_cubic?
                        cur_array,cur_qs,edge_qs,cur_us,_,lsd=msc2d(cur_array,error_bound,alpha,beta,9999,args.max_step,args.anchor_rate,rate_list=None,x_preded=False,y_preded=False,\
                                                sz3_interp=args.sz_interp,multidim_level=args.multidim_level,lorenzo=-1,sample_rate=0.0,min_sampled_points=100,random_access=False,verbose=False,fix_algo=args.fix_algo,fix_algo_list=None)
                            #print(len(cur_qs[max_level]))
                            #print(len(test_qs[max_level]))
                        for level in range(max_level+1):
                                #print(level)
                            test_qs[level]+=cur_qs[level]
                        test_us+=cur_us
                            #zero_square_error=np.sum((array[x_start:x_end,y_start:y_end]-themean*np.ones((max_step+1,max_step+1)) )**2)
                        square_error+=np.sum((array[x_start:x_end,y_start:y_end]-cur_array)**2)
                            
                        element_counts+=(max_step+1)**2 
                t_mse=square_error/element_counts
                    #zero_mse=zero_square_error/element_counts
                psnr=20*math.log(rng,10)-10*math.log(t_mse,10)
                    #zero_psnr=20*math.log(themax-themin,10)-10*math.log(zero_mse,10)
                    #print(zero_psnr)
                  
                np.array(sum(test_qs,[]),dtype=np.int32).tofile(tq_name)
                np.array(sum(test_us,[]),dtype=np.int32).tofile(tu_name)
                with os.popen("sz_backend %s %s" % (tq_name,tu_name)) as f:
                    lines=f.read().splitlines()
                    cr=eval(lines[4].split("=")[-1])
                    if args.anchor_rate==0:
                        anchor_ratio=1/(args.max_step**2)
                        cr=1/((1-anchor_ratio)/cr+anchor_ratio)
                    bitrate=32/cr
                os.system("rm -f %s;rm -f %s" % (tq_name,tu_name))
                    #pdb=(psnr-zero_psnr)/bitrate
                if psnr<=bestp and bitrate>=bestb:
                    continue
                elif psnr>=bestp and bitrate<=bestb:

                    bestalpha=alpha
                    bestbeta=beta
                       
                    bestb=bitrate
                    bestp=psnr
                           
                else:
                    if psnr>bestp:
                        new_error_bound=1.2*error_bound
                    else:
                        new_error_bound=0.8*error_bound
                    test_qs=[[] for i in range(max_level+1)]
                    test_us=[]
                    square_error=0
                    #zero_square_error=0
                    element_counts=0
                    themax=-9999999999999
                    themin=99999999999999
                        #themean=0
                        #print(themean)
                    for i in range(0,block_num_x,steplength):
                        for j in range(0,block_num_y,steplength):
                              
                            x_start=max_step*i
                            y_start=max_step*j
                            x_end=x_start+max_step+1
                            y_end=y_start+max_step+1
                                #print(x_start)
                                #print(y_start)
                            cur_array=np.copy(array[x_start:x_end,y_start:y_end])
                                
                            cur_array,cur_qs,edge_qs,cur_us,_,lsd=msc2d(cur_array,new_error_bound,alpha,beta,9999,args.max_step,args.anchor_rate,rate_list=None,x_preded=False,y_preded=False,\
                                                    sz3_interp=args.sz_interp,multidim_level=args.multidim_level,lorenzo=-1,sample_rate=0.0,min_sampled_points=100,random_access=False,verbose=False,fix_algo=args.fix_algo,fix_algo_list=None)
                                #print(len(cur_qs[max_level]))
                                #print(len(test_qs[max_level]))
                            for level in range(max_level+1):
                                    #print(level)
                                test_qs[level]+=cur_qs[level]
                            test_us+=cur_us
                                #zero_square_error=np.sum((array[x_start:x_end,y_start:y_end]-themean*np.ones((max_step+1,max_step+1)) )**2)
                            square_error+=np.sum((array[x_start:x_end,y_start:y_end]-cur_array)**2)
                                
                            element_counts+=(max_step+1)**2 
                    t_mse=square_error/element_counts
                        #zero_mse=zero_square_error/element_counts
                    psnr_r=20*math.log(rng,10)-10*math.log(t_mse,10)
                        #zero_psnr=20*math.log(themax-themin,10)-10*math.log(zero_mse,10)
                        #print(zero_psnr)
                      
                    np.array(sum(test_qs,[]),dtype=np.int32).tofile(tq_name)
                    np.array(sum(test_us,[]),dtype=np.int32).tofile(tu_name)
                    with os.popen("sz_backend %s %s" % (tq_name,tu_name)) as f:
                        lines=f.read().splitlines()
                        cr=eval(lines[4].split("=")[-1])
                        if args.anchor_rate==0:
                            anchor_ratio=1/(args.max_step**2)
                            cr=1/((1-anchor_ratio)/cr+anchor_ratio)
                        bitrate_r=32/cr
                    os.system("rm -f %s;rm -f %s" % (tq_name,tu_name))
                    a=(psnr-psnr_r)/(bitrate-bitrate_r)
                    b=psnr-a*bitrate
                        #print(a)
                        #print(b)
                    reg=a*bestb+b
                    if reg>bestp:
                        bestalpha=alpha
                        bestbeta=beta
                       
                        bestb=bitrate
                        bestp=psnr
                if alpha**(max_level-1)<=beta:
                    break

                    
                    
                   


        print("Autotuning finished. Selected alpha: %f. Selected beta: %f. Best bitrate: %f. Best PSNR: %f."\
        %(bestalpha,bestbeta,bestb,bestp) )
        args.rate=bestalpha
        args.maximum_rate=bestbeta

    else:       
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


    last_x=((size_x-1)//max_step)*max_step
    last_y=((size_y-1)//max_step)*max_step   

    #il_count=0
    #ic_count=0
    #im_count=0
    #l_count=0
    lorenzo_level=args.lorenzo_fallback_check
    lorenzo_sample_ratio=args.fallback_sample_ratio
    #currently no coeff and levelwise predictor selection.
    for x_start in range(0,last_x,max_step):
        for y_start in range(0,last_y,max_step):
            print(x_start,y_start)
            x_end=size_x-1 if x_start==last_x-max_step else x_start+max_step 
            y_end=size_y-1 if y_start==last_y-max_step else y_start+max_step 
            array[x_start:x_end+1,y_start:y_end+1],cur_qs,cur_lorenzo_qs,cur_us,cur_selected,lsd=\
            msc2d(array[x_start:x_end+1,y_start:y_end+1],error_bound,rate,maximum_rate,min_coeff_level,max_step,anchor_rate,\
                rate_list=rate_list,sz3_interp=args.sz_interp,multidim_level=args.multidim_level,lorenzo=args.lorenzo_fallback_check,\
                sample_rate=args.fallback_sample_ratio,min_sampled_points=10,x_preded=(x_start>0),y_preded=(y_start>0),random_access=False,fix_algo=args.fix_algo)

            for i in range(max_level+1):
                #print(len(cur_qs[i]))
                qs[i]+=cur_qs[i]

            us+=cur_us
            lorenzo_qs+=cur_lorenzo_qs
            #if "lorenzo" in cur_selected[-1]:
                #print(x_start,y_start)



     






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