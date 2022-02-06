from torch.utils.data import Dataset
import numpy as np
import os


class CESM(Dataset):
    def __init__(self,path,start,end,level=2,ratio=10,field='CLDHGH',global_max=None,global_min=None,epsilon=-1):
        height=1800
        width=3600
        blocks=[]
        regs=[]
        path=os.path.join(path,field)
        size=level**2-1
        for i in range(start,end):
            s=str(i)
            if i<10:
                s="0"+s
            filename="%s_%s.dat" % (field,s)
            if field=="PHIS":
                filename+=".log10"
            filepath=os.path.join(path,filename)
            array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        #print(array)
            for x in range(0,height,size):
                for y in range(0,width,size):
                    if np.random.choice(ratio)>0:
                        continue
                    block=array[x-level+1:x+1,y-level+1:y+1].flatten()
                    
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
        self.picts=np.array(picts)
    def __len__(self):
        return self.blocks.shape[0]
    def __getitem__(self,idx):
        return self.blocks[idx],self.regs[idx]