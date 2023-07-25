import numpy as np
import os
import sys
#cropsize=480
imagesize=128
ipath=sys.argv[1]
ifile=os.path.basename(ipath).split('.')[0]
outfolder=sys.argv[2]

dims=[]
for i in range(3,len(sys.argv)):
    dims.append(int(sys.argv[i]))
dims=tuple(dims)
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
img_idx=0
   
filename="%s" %(ipath)
arr=np.fromfile(filename,dtype=np.float32).reshape(dims)

for x in range(0,dims[0]):
    cur_slice=arr[x,:,:]
    cur_filename=filename+"_%d_%d_%d.dat"%(img_idx,dims[1],dims[2])
    img_idx+=1
    cur_path=os.path.join(outfolder,cur_filename)
    cur_slice.tofile(cur_path)

for y in range(0,dims[1]):
    cur_slice=arr[:,y,:]
    cur_filename=filename+"_%d_%d_%d.dat"%(img_idx,dims[0],dims[2])
    img_idx+=1
    cur_path=os.path.join(outfolder,cur_filename)
    cur_slice.tofile(cur_path)


for z in range(0,dims[2]):
    cur_slice=arr[:,:,z]
    cur_filename=filename+"_%d_%d_%d.dat"%(img_idx,dims[0],dims[1])
    img_idx+=1
    cur_path=os.path.join(outfolder,cur_filename)
    cur_slice.tofile(cur_path)