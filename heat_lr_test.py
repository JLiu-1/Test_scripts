import numpy as np 
import sys
import os

input_folder=sys.argv[1]
ckpt_file=sys.argv[2]
start=int(sys.argv[3])
end=int(sys.argv[4])
size_x=200
size_y=200
average_mse=0
ckpt=np.fromfile(ckpt_file)
for i in range(start,end):
	filename_x="%d.dat" % (i) 
    filename_y="%d.dat" % (i+1) 
           
    filepath_x=os.path.join(input_folder,filename_x)
    filepath_y=os.path.join(input_folder,filename_y)
    array_x=np.fromfile(filepath_x,dtype=np.float32).reshape((size_x,size_y))
    array_y=np.fromfile(filepath_y,dtype=np.float32).reshape((size_x,size_y))
    array_predict=np.array(array_y,copy=True)
    for x in range(1,size_x-1):
        for y in range(1,size_y-1):
            block=array_x[x-1:x+2,y-1:y+2].flatten()
            reg=array_y[x][y]