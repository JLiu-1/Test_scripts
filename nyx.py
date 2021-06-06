from torch.utils.data import Dataset
import numpy as np
import os
class NYX(Dataset):
    def __init__(self,path,field,start,end,log=0,global_max=None,global_min=None,norm_min=-1,epsilon=-1):
        size_x=512
        size_y=512
        size_z=512
        blocks=[]
        regs=[]
       # count=[0,0,0,0]
        for i in range(start,end):
            s=str(i)
            
            filename="%s_%s.dat" % (field,s) 
            if log:
                filename+=".log10"
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((size_x,size_y,size_z))
        #print(array)
            for x in range(1,size_x):
                for y in range(1,size_y):
                    for z in range(1,size_z):
                        block=array[x-1:x+1,y-1:y+1,z-1:z+1].flatten()
                        
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
                        blocks.append(block[:7])
                    #print(array[x:x+size,y:y+size])
                        regs.append(block[7])
        #print(count)
        self.blocks=np.array(blocks,dtype=np.float32)
        self.regs=np.array(regs,dtype=np.float32)
        print(self.blocks.shape[0])

        
    def __len__(self):
        return self.blocks.shape[0]
    def __getitem__(self,idx):
        return self.blocks[idx],[self.regs[idx]]
