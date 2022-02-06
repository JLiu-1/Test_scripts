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

array=np.fromfile(args.input,dtype=np.float32).reshape((1800,3600))
coefs=np.fromfile(args.checkpoint,dtype=np.float64)
error_bound=args.error*(np.max(array)-np.min(array))

qs=[]
us=[]
size=level**2-1
for x in range(1800):
    for y in range(3600):
        
        orig=array[x][y]
        if not (x>=level-1 and y>=level-1 ):
            f_01=array[x-1][y] if x else 0
            f_10=array[x][y-1] if y else 0
            
            f_00=array[x-1][y-1] if x and y else 0
                
            pred=f_01+f_10-f_00
                
        else:
            block=array[x-level+1:x+1,y-level+1:y+1].flatten()[:size]
                
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