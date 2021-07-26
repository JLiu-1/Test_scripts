from torch.utils.data import Dataset
import numpy as np
import os
class Miranda_cubic(Dataset):
    def __init__(self,path,start,end,ratio=10,global_max=None,global_min=None,norm_min=-1,epsilon=-1):
        size_x=256
        size_y=384
        size_z=384
        blocks=[]
        regs=[]
       # count=[0,0,0,0]
        filelist=sorted(os.listdir(path))
        for i in range(start,end):
            
            
            filename=filelist[i]
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((size_x,size_y,size_z))
        #print(array)
            for x in range(1,size_x):
                for y in range(1,size_y):
                    for z in range(1,size_z):
                        
                        if x%2==0 and y%2==0 and z%2==0:
                            continue

                        for i in range(3):
                            if i!=0:
                                continue
                            if i==0 and (x-3<0 or x+3>=size_x):
                                continue 
                            if i==1 and (y-3<0 or y+3>=size_y):
                                continue 
                            if i==2 and (z-3<0 or z+3>=size_z):
                                continue 
                            if np.random.choice(ratio)>0:
                                continue
                            block=np.zeros((4,),dtype=np.float32)
                            reg=array[x][y][z]
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
                            rng=np.max(block)-np.min(block)
                            if rng<=epsilon:
                                continue
                        
                            if global_max!=None:
                               
                                if norm_min==0:
                                    block=(block-global_min)/(global_max-global_min)
                                else:
                                    block=(block-global_min)*2/(global_max-global_min)-1
                            blocks.append(block)
                    #print(array[x:x+size,y:y+size])
                            regs.append(reg)
        #print(count)
        self.blocks=np.array(blocks,dtype=np.float32)
        self.regs=np.array(regs,dtype=np.float32)
        self.regs=np.expand_dims(self.regs,axis=-1)
        print(self.blocks.shape[0])

        
    def __len__(self):
        return self.blocks.shape[0]
    def __getitem__(self,idx):
        return self.blocks[idx],self.regs[idx]
