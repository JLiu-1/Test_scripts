import numpy as np 
import sys
import os
import argparse
import torch
import torch.nn as nn

checkpoint=sys.argv[1]
output=sys.argv[2]
statedict=torch.load(checkpoint,map_location='cpu')["state_dict"]
array=np.zeros(8,dtype=np.float32)
count=0
for key in statedict:
    print(statedict[key])
    if count==0:

        array[0:4]=statedict[key].detach().numpy()
        count+=1
   
print(array)
array.tofile(output)

