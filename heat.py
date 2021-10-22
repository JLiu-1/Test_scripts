from torch.utils.data import Dataset
import numpy as np
import os
class Heat(Dataset):
    def __init__(self,path,start,end,size_x,size_y,stride=1,ratio=1,global_max=None,global_min=None,norm_min=-1,epsilon=-1):
        blocks=[]
        regs=[]
       # count=[0,0,0,0]

        for i in range(start,end):
           
            
            filename_x="%d.dat" % (i) 
            filename_y="%d.dat" % (i+1) 
           
            filepath_x=os.path.join(path,filename_x)
            filepath_y=os.path.join(path,filename_y)
            array_x=np.fromfile(filepath_x,dtype=np.float32).reshape((size_x,size_y))
            array_y=np.fromfile(filepath_y,dtype=np.float32).reshape((size_x,size_y))
        #print(array)
            for x in range(1,size_x-1,stride):
                for y in range(1,size_y-1,stride):
                    
                    if np.random.choice(ratio)>0:
                        continue
                    block=array_x[x-1:x+2,y-1:y+2].flatten()
                    reg=array_y[x][y]
                    if global_max!=None:
                        rng=global_max-global_min
                        if np.max(block)>global_max or np.min(block)<global_min:
                            continue
                        if epsilon>0:
                            r=np.max(block)-np.min(block)
                            
                            if r<rng*epsilon:
                                continue
                        if norm_min==0:
                            block=(block-global_min)/(global_max-global_min)
                            reg=(reg-global_min)/(global_max-global_min)
                        else:
                            block=(block-global_min)*2/(global_max-global_min)-1
                            reg=(reg-global_min)*2/(global_max-global_min)-1
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


class Heat_Double(Dataset):
    def __init__(self,path,start,end,size_x,size_y,ratio=10,global_max=None,global_min=None,norm_min=-1,epsilon=-1):
        blocks=[]
        regs=[]
       # count=[0,0,0,0]

        for i in range(start,end):
           
            
            filename_x="%d.dat" % (i) 
            filename_y="%d.dat" % (i+1) 
           
            filepath_x=os.path.join(path,filename_x)
            filepath_y=os.path.join(path,filename_y)
            array_x=np.fromfile(filepath_x,dtype=np.float32).reshape((size_x,size_y))
            array_y=np.fromfile(filepath_y,dtype=np.float32).reshape((size_x,size_y))
        #print(array)
            for x in range(1,size_x-1):
                for y in range(1,size_y-1):
                    
                    if np.random.choice(ratio)>0:
                        continue
                    block=array_x[x-1:x+2,y-1:y+2].flatten()
                    reg=array_y[x][y]
                    if global_max!=None:
                        rng=global_max-global_min
                        if epsilon>0:
                            r=np.max(block)-np.min(block)
                            
                            if r<rng*epsilon:
                                continue
                        if norm_min==0:
                            block=(block-global_min)/(global_max-global_min)
                            reg=(reg-global_min)/(global_max-global_min)
                        else:
                            block=(block-global_min)*2/(global_max-global_min)-1
                            reg=(reg-global_min)*2/(global_max-global_min)-1
                    blocks.append(block.astype(np.double))
                    #print(array[x:x+size,y:y+size])
                    regs.append(reg)
        #print(count)
        self.blocks=np.array(blocks,dtype=np.double)
        self.regs=np.array(regs,dtype=np.double)
        self.regs=np.expand_dims(self.regs,axis=-1)
        print(self.blocks.shape[0])

        
    def __len__(self):
        return self.blocks.shape[0]
    def __getitem__(self,idx):
        return self.blocks[idx],self.regs[idx]        

class Heat_Random(Dataset):
    def __init__(self,size):
        blocks=[]
        regs=[]
        for i in range(size):
            block=np.random.rand(3,3)
            reg=(block[0,1]+block[1,0]+block[1,2]+block[2,1])/4
            block=block.flatten()
            #print(block)
            #print(reg)
            blocks.append(block)
                   
            regs.append(reg)
       
        self.blocks=np.array(blocks,dtype=np.float32)
        self.regs=np.array(regs,dtype=np.float32)
        self.regs=np.expand_dims(self.regs,axis=-1)
        print(self.blocks.shape[0])

        
    def __len__(self):
        return self.blocks.shape[0]
    def __getitem__(self,idx):
        return self.blocks[idx],self.regs[idx]


class Heat_Double_Random(Dataset):
    def __init__(self,size):
        blocks=[]
        regs=[]
        for i in range(size):
            block=np.random.rand(3,3)
            reg=(block[0,1]+block[1,0]+block[1,2]+block[2,1])/4
        
            blocks.append(block)
                   
            regs.append(reg)
       
        self.blocks=np.array(blocks,dtype=np.double)
        self.regs=np.array(regs,dtype=np.double)
        self.regs=np.expand_dims(self.regs,axis=-1)
        print(self.blocks.shape[0])

        
    def __len__(self):
        return self.blocks.shape[0]
    def __getitem__(self,idx):
        return self.blocks[idx],self.regs[idx]        
