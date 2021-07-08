import numpy as np 
import sys

array=np.fromfile(sys.argv[1],dtype=np.float32)
size_x=int(sys.argv[2])
size_y=int(sys.argv[3])
size_z=int(sys.argv[4])
coefs=np.fromfile(sys.argv[5],dtype=np.float32)
array=array.reshape((size_x,size_y,size_z))
loss1=0
loss2=0
for x in range(1,size_x):
    for y in range(1,size_y):
        for z in range(1,size_z):
                        
            if x%2==0 and y%2==0 and z%2==0:
                continue
            for i in range(3):
                if i==0 and (x-3<0 or x+3>=size_x):
                    continue 
                if i==1 and (y-3<0 or y+3>=size_y):
                    continue 
                if i==2 and (z-3<0 or z+3>=size_z):
                    continue 
                            
                block=np.zeros((4,),dtype=np.float32)
                target=array[x][y][z]
                if i==0:
                    block[0]=array[x-3][y][z]
                    block[1]=array[x-1][y][z]
                    block[2]=array[x+1][y][z]
                    block[3]=array[x+3][y][z]
                elif i==1:
                    block[0]=array[x][y-3][z]
                    block[1]=array[x][y-1][z]
                    block[2]=array[x][y+1][z]
                    block[3]=array[x][y+3][z]
                else:
                    block[0]=array[x][y][z-3]
                    block[1]=array[x][y][z-1]
                    block[2]=array[x][y][z+1]
                    block[3]=array[x][y][z+3]
                pred1=(-block[0]+9*block[1]+9*block[2]-block[3])/16
                pred2=0
                for i in range(4):
                    pred2+=coefs[i]*block[i]
                loss1+=(pred1-target)**2
                loss2+=(pred2-target)**2


print(loss1)
print(loss2)

