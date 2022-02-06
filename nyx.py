from torch.utils.data import Dataset
import numpy as np
import os
class NYX(Dataset):
    def __init__(self,path,field,start,end,level=2,ratio=10,log=0,global_max=None,global_min=None,norm_min=-1,epsilon=-1):
        size_x=512
        size_y=512
        size_z=512
        blocks=[]
        regs=[]
       # count=[0,0,0,0]

        for i in range(start,end):
            s=str(i)
            if i>=0:
                filename="%s_%s.dat" % (field,s) 
            else:
                filename="%s.dat" % field
            if log:
                filename+=".log10"
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((size_x,size_y,size_z))
            size=level**3-1
        #print(array)
            for x in range(level-1,size_x):
                for y in range(level-1,size_y):
                    for z in range(level-1,size_z):
                        if np.random.choice(ratio)>0:
                            continue
                        block=array[x-level+1:x+1,y-level+1:y+1,z-level+1:z+1].flatten()
                        
                        if global_max!=None:
                            rng=global_max-global_min
                            if epsilon>0:
                                r=np.max(block)-np.min(block)
                            
                                if r<rng*epsilon:
                                    continue
                            if norm_min==0:
                                block=(block-global_min)/(global_max-global_min)
                            else:
                                block=(block-global_min)*2/(global_max-global_min)-1
                        blocks.append(block[:size])
                    #print(array[x:x+size,y:y+size])
                        regs.append(block[size])
        #print(count)
        self.blocks=np.array(blocks,dtype=np.float32)
        self.regs=np.array(regs,dtype=np.float32)
        self.regs=np.expand_dims(self.regs,axis=-1)
        print(self.blocks.shape[0])

        
    def __len__(self):
        return self.blocks.shape[0]
    def __getitem__(self,idx):
        return self.blocks[idx],self.regs[idx]
