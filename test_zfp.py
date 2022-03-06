import os
import numpy as np 
import argparse
import pandas as pd
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--input','-i',type=str)
    parser.add_argument('--output','-o',type=str)
    
   
    
    parser.add_argument('--dim','-d',type=int,default=2)
    parser.add_argument('--dims','-m',type=str,nargs="+")
    #parser.add_argument('--config','-c',type=str,default=None)
    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)
    

    args = parser.parse_args()
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    num_files=len(datafiles)

    #ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-3 for i in range(10,21,5)]
    ebs=[1e-4,1e-3,1e-2]
    num_ebs=len(ebs)

    cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    #maxpwerr=np.zeros((len(ebs)+1,len(idxlist)+1),dtype=np.float32)
    #nrmse=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    #algo=np.zeros((num_ebs,num_files),dtype=np.int32)
    pid=os.getpid()
    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            
            filepath=os.path.join(datafolder,datafile)
           
            
            arr=np.fromfile(filepath,dtype=np.float32)
            rng=np.max(arr)-np.min(arr)
            abseb=rng*eb
            comm="zfp -s -i %s -z %s.out -f -%d %s -a %f &>%s.txt" % (filepath,pid,args.dim," ".join(args.dims),eb,pid)
            print(comm)
            os.system(comm)
            with open("%s.txt"%pid,"r") as f:
                lines=f.read().splitlines()
                #print(lines)
                r=eval(lines[-5].split(' ')[7].split("=")[-1])
                try:
                    p=eval(lines[-1].split(' ')[-1])
                except:
                    p=np.inf
                n=eval(lines[-5].split(' ')[10].split("=")[-1])
                cr[i][j]=r 
                psnr[i][j]=p
                overall_psnr[i]+=n**2
            os.system("rm -f %s.dat;rm -f %s.txt" % (pid,pid))
            print(datafile,eb,rng,abseb,r,p)

    overall_psnr=overall_psnr/num_files
    overall_psnr=np.sqrt(overall_psnr)
    overall_psnr=-20*np.log10(overall_psnr)
    overall_cr=np.reciprocal(np.mean(np.reciprocal(cr),axis=1))


    cr_df=pd.DataFrame(cr,index=ebs,columns=datafiles)
    psnr_df=pd.DataFrame(psnr,index=ebs,columns=datafiles)
    overall_cr_df=pd.DataFrame(overall_cr,index=ebs,columns=["overall_cr"])
    overall_psnr_df=pd.DataFrame(overall_psnr,index=ebs,columns=["overall_psnr"])
    
    cr_df.to_csv("%s_cr.tsv" % args.output,sep='\t')
    psnr_df.to_csv("%s_psnr.tsv" % args.output,sep='\t')
    overall_cr_df.to_csv("%s_overall_cr.tsv" % args.output,sep='\t')
    overall_psnr_df.to_csv("%s_overall_psnr.tsv" % args.output,sep='\t')
   
               
        
     

        
   


