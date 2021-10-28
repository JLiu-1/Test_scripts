import numpy as np 

import os
import argparse
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
#parser.add_argument('--dropout','-d',type=float,default=0)
parser.add_argument('--actv','-a',type=str,default='no')
parser.add_argument('--checkpoint','-c',type=str)
parser.add_argument('--double','-d',type=int,default=0)
parser.add_argument('--conv','-v',type=int,default=0)

args = parser.parse_args()

actv_dict={"no":nn.Identity,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh}
actv=actv_dict[args.actv]
if args.conv:
    layer=nn.Conv2d(1,1,3,bias=False)
else:
    layer=nn.Linear(9,1)
model=nn.Sequential(layer,actv())
if args.double:
    model=model.double()
model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
model.eval()

for name,parameters in model.named_parameters():
    print(name)
    a=parameters.detach().numpy()
    print(a)
    print(np.sum(a))
#x=torch.tensor([[1.0,1,1,1,1,1,1,1,1]])
#y=model(x)
#print(y)