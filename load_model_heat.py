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


args = parser.parse_args()

actv_dict={"no":nn.Identity,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh}
actv=actv_dict[args.actv]
model=nn.Sequential(nn.Linear(9,1),actv())
if args.double:
    model=model.double()
model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
model.eval()
for name,parameters in model.named_parameters():
    print(name)
    print(parameters.detach().numpy())