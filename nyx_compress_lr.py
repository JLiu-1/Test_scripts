import numpy as np 

import os
import argparse
import torch
import torch.nn as nn
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
parser.add_argument('--checkpoint','-c',type=str)
#parser.add_argument('--norm_max','-nx',type=float,default=1)
#parser.add_argument('--norm_min','-ni',type=float,default=-1)
parser.add_argument('--max','-mx',type=float,
                   default=5.06394195556640625)
parser.add_argument('--min','-mi',type=float,
                   default=-1.306397557258605957)


args = parser.parse_args()

#actv_dict={"no":nn.Identity,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh}
#actv=actv_dict[args.actv]
#model=nn.Sequential(nn.Linear(7,1),actv())
#model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
#model.eval()
#for name,parameters in model.named_parameters():
#    print(name)
#    print(parameters.detach().numpy())

array=np.fromfile(args.input,dtype=np.float32).reshape((512,512,512))
coefs=np.fromfile(args.checkpoint,dtype=np.float64)
error_bound=args.error*(np.max(array)-np.min(array))

qs=[]
us=[]
for x in range(512):
    for y in range(512):
        for z in range(512):
            orig=array[x][y][z]
            if not (x and y and z):
                f_011=array[x-1][y][z] if x else 0
                f_101=array[x][y-1][z] if y else 0
                f_110=array[x][y][z-1] if z else 0
                f_001=array[x-1][y-1][z] if x and y else 0
                f_100=array[x][y-1][z-1] if y and z else 0
                f_010=array[x-1][y][z-1] if x and z else 0
                f_000=array[x-1][y-1][z-1] if x and y and z else 0
                
                pred=f_000+f_011+f_101+f_110-f_001-f_010-f_100
                
            else:
                block=array[x-1:x+1,y-1:y+1,z-1:z+1].flatten()[:7]
                #block=(block-args.min)/(args.max-args.min)
                #block=block*(args.norm_max-args.norm_min)+args.norm_min
                #block=np.expand_dims(block,axis=0)
                pred=block*coefs
                #pred=(pred-args.norm_min)/(args.norm_max-args.norm_min)
                #pred=pred*(args.max-args.min)+args.min
                #print(pred)
            q,decomp=quantize(orig,pred,error_bound)
            qs.append(q)
            if q==0:
                us.append(decomp)
            array[x][y][z]=decomp

quants=np.array(qs,dtype=np.int32)
unpreds=np.array(us,dtype=np.float32)
array.tofile(args.output)
quants.tofile("q.dat")
unpreds.tofile("u.dat")