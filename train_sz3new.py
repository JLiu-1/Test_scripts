import os
import numpy as np 
import argparse
import pandas as pd
from collections import Counter
def find_most_occured(l):
    c=Counter()
    for e in l:
        c[e]+=1
    moe=None
    mot=0
    for e in l:
        if c[e]>mot:
            mot=c[e]
            moe=e
    return moe

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--input','-i',type=str)
    parser.add_argument('--output','-o',type=str)
    
   
    
    parser.add_argument('--dim','-d',type=int,default=2)
    parser.add_argument('--lorenzo','-z',type=int,default=1)
    parser.add_argument('--dims','-m',type=str,nargs="+")
    parser.add_argument('--levelwise','-l',type=int,default=0)
    parser.add_argument('--maxstep','-s',type=int,default=0)
    parser.add_argument('--blocksize','-b',type=int,default=0)
    
    #parser.add_argument('--abtuningrate',"-a",type=float,default=0.01)
    parser.add_argument('--field',"-f",type=str,default=None)
    parser.add_argument('--predtuningrate',"-p",type=float,default=0.01)
    #parser.add_argument('--totaltuningrate',"-t",type=float,default=None)
    #parser.add_argument('--cr_tuning',"-c",type=int,default=0)
    #parser.add_argument('--linear_reduce',"-r",type=int,default=0)
    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)

    alpha_list=[i/8 for i in range(8,21)]
    beta_list=[i/4 for i in range(4,21)]
    

    args = parser.parse_args() 
    
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    if args.field!=None:
        datafiles=[f for f in datafiles if args.field in f]
    num_files=len(datafiles)

    #ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-3 for i in range(10,21,5)]








    ebs=[1e-4,1e-3,1e-2]
    num_ebs=len(ebs)
    if args.blocksize>0:
        blocksize=args.blocksize
        algo="ALGO_INTERP_BLOCKED"
    else:
        blocksize=32 
        algo="ALGO_INTERP_LORENZO"


   
    #cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    #psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    alphas=np.zeros((num_ebs,num_files),dtype=np.float32)
    betas=np.zeros((num_ebs,num_files),dtype=np.float32)
    most_alphas=np.zeros((num_ebs),dtype=np.float32)
    most_betas=np.zeros((num_ebs),dtype=np.float32)
    #overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    #overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    pid=os.getpid()
    
    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            filepath=os.path.join(datafolder,datafile)
            bestalpha=1
            bestbeta=1
               
            bestb=9999
                
            bestp=0

            for alpha in alpha_list:
                for beta in beta_list:
    
                    configstr="[GlobalSettings]\nCmprAlgo = %s \n[AlgoSettings]\nalpha = %f \nbeta = %f \npredictorTuningRate= %f \nlevelwisePredictionSelection = %d \nmaxStep = %d \ninterpolationBlockSize = %d \ntestLorenzo= %d \ntrain= 1 \n" % \
                    (algo,alpha,beta,args.predtuningrate,args.levelwise,args.maxstep,blocksize,args.lorenzo) 
                    with open("%s.config" % pid,"w") as f:
                        f.write(configstr)


    
            
            

            
                    comm="sz3_new -z -f -a -i %s -o %s.out -M REL %f -%d %s -c %s.config" % (filepath,pid,eb,args.dim," ".join(args.dims),pid)
                
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        print(lines)
                        
                        cr=eval(lines[-3].split('=')[-1])
                        bitrate=32/cr
                        psnr=eval(lines[-6].split(',')[0].split('=')[-1])
                    comm="rm -f %s.out" % pid
                    os.system(comm)
                    if psnr<=bestp and bitrate>=bestb:
                        
                        continue
                    elif psnr>=bestp and bitrate<=bestb:

                        bestalpha=alpha
                        bestbeta=beta
                   
                        bestb=bitrate
                        bestp=psnr
                    else:
                        if psnr>bestp:
                            new_eb=1.2*eb
                        else:
                            new_eb=0.8*eb
                        comm="sz3_new -z -f -a -i %s -o %s.out -M REL %f -%d %s -c %s.config" % (filepath,pid,new_eb,args.dim," ".join(args.dims),pid)
                
                        with os.popen(comm) as f:
                            lines=f.read().splitlines()
                            #print(lines)
                            
                            cr=eval(lines[-3].split('=')[-1])
                            bitrate_r=32/cr
                            psnr_r=eval(lines[-6].split(',')[0].split('=')[-1])
                        a=(psnr-psnr_r)/(bitrate-bitrate_r+1e-12)
                        b=psnr-a*bitrate
                        #print(a)
                        #print(b)
                        reg=a*bestb+b
                        if reg>bestp:
                            bestalpha=alpha
                            bestbeta=beta
                       
                            bestb=bitrate
                            bestp=psnr
                    os.system("rm -f %s.config" % pid)
            print(eb,filepath,bestalpha,bestbeta)
            alphas[i][j]=bestalpha
            betas[i][j]=bestbeta


                 
                        
                       

                                
            
                
                

            
                    




                    








                    
    ave_alphas=np.mean(alphas,axis=1)
    ave_betas=np.mead(betas,axis=1)
    for i in range(num_ebs):
        most_alphas[i]=find_most_occured(alphas[i])
        most_betas[i]=find_most_occured(betas[i])


    alphas_df=pd.DataFrame(alphas,index=ebs,columns=datafiles)
    betas_df=pd.DataFrame(betas,index=ebs,columns=datafiles)
    ave_alphas_df=pd.DataFrame(ave_alphas,index=ebs,columns="ave")
    ave_betas_df=pd.DataFrame(ave_betas,index=ebs,columns="ave")
    most_alphas_df=pd.DataFrame(most_alphas,index=ebs,columns="most")
    most_betas_df=pd.DataFrame(most_betas,index=ebs,columns="most")


    
    alphas_df.to_csv("%s_alphas.tsv" % args.output,sep='\t')
    betas_df.to_csv("%s_betas.tsv" % args.output,sep='\t')
    ave_alphas_df.to_csv("%s_avealphas.tsv" % args.output,sep='\t')
    ave_betas_df.to_csv("%s_avebetas.tsv" % args.output,sep='\t')
    most_alphas_df.to_csv("%s_mostalphas.tsv" % args.output,sep='\t')
    most_betas_df.to_csv("%s_mostbetas.tsv" % args.output,sep='\t')