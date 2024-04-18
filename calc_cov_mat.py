import sys
import numpy as np 
from sklearn.preprocessing import scale

cur_arg=1
ifile=sys.argv[cur_arg]
cur_arg+=1
dim_count=eval(sys.argv[cur_arg])
cur_arg+=1
dims=[]
for _ in range(dim_count):
    dims.append(eval(sys.argv[cur_arg]))
    cur_arg+=1
if dim_count==3:
    slice_dim=eval(sys.argv[cur_arg])
    cur_arg+=1
    slice_idx=eval(sys.argv[cur_arg])
    cur_arg+=1
r_or_c=eval(sys.argv[cur_arg])
cur_arg+=1
ofile=sys.argv[cur_arg]
cur_arg+=1

input_array=np.fromfile(ifile,dtype=np.float32).reshape(dims)
if dim_count==3:
    input_array=np.take(input_array,slice_idx,slice_dim)

mat_dims=input_array.shape

if r_or_c: #c
    input_array=scale(input_array,axis=0)
    output_array=np.matmul(np.transpose(input_array),input_array)/(mat_dims[0]-1)
else:
    input_array=scale(input_array,axis=1)
    output_array=np.matmul(input_array,np.transpose(input_array))/(mat_dims[1]-1)

output_array.tofile(ofile)