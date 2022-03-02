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
    parser.add_argument('--levelwise','-l',type=int,default=0)
    parser.add_argument('--maxstep','-s',type=int,default=0)
    parser.add_argument('--blocksize','-b',type=int,default=0)
    parser.add_argument('--abtuningrate',"-a",type=float,default=0.01)
    parser.add_argument('--predtuningrate',"-p",type=float,default=0.01)
    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)


    

    args = parser.parse_args()
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]!="txt" and file.split(".")[-1]!="out" and file.split(".")[-1]!="config"]
    num_files=len(datafiles)

    #ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-2 for i in range(1,11)]
    ebs=[1e-4,1e-3,1e-2]
    num_ebs=len(ebs)
    if args.blocksize>0:
        blocksize=blocksize
        algo="ALGO_INTERP_BLOCKED"
    else:
        blocksize=32 
        algo="ALGO_INTERP_LORENZO"
    cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    alpha=np.zeros((num_ebs,num_files),dtype=np.float32)
    beta=np.zeros((num_ebs,num_files),dtype=np.float32)
    pid=os.getpid()
    
    configstr="[GlobalSettings]\nCmprAlgo = %s \n \
    [AlgoSettings]\nautoTuningRate = %f \n predictorTuningRate= %f \n levelwisePredictionSelection = %d \n \
    maxStep = %d \n interpolationBlockSize = %d \n" % (algo,args.abtuningrate,args.predtuningrate,args.levelwise,args.maxstep,blocksize) 
    with open("%s.config" % pid,"w") as f:
        f.write(configstr)


    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            
            filepath=os.path.join(datafolder,datafile)

            
            comm="sz -z -f -a -i %s -o %s.out -M REL %f -%d %s -c %s.config" % (filepath,pid,eb,args.dim," ".join(args.dims),pid)
            
            with os.popen(comm) as f:
                lines=f.read().splitlines()
                r=eval(lines[-3].split('=')[-1])
                p=eval(lines[-6].split(',')[0].split('=')[-1])
                cr[i][j]=r 
                psnr[i][j]=p
            
                
                

            
            comm="rm -f %s.out;rm -f %s.config" % (pid,pid)
            os.system(comm)
            

    cr_df=pd.DataFrame(cr,index=ebs,columns=datafiles)
    psnr_df=pd.DataFrame(psnr,index=ebs,columns=datafiles)
    cr_df.to_csv("%s_cr.tsv" % args.output,sep='\t')
    psnr_df.to_csv("%s_psnr.tsv" % args.output,sep='\t')