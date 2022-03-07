import os
import numpy as np 
import argparse
import pandas as pd
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
    
    parser.add_argument('--abtuningrate',"-a",type=float,default=0.01)
    parser.add_argument('--predtuningrate',"-p",type=float,default=0.01)
    parser.add_argument('--totaltuningrate',"-t",type=float,default=None)
    parser.add_argument('--cr_tuning',"-c",type=int,default=0)
    parser.add_argument('--linear_reduce',"-r",type=int,default=0)
    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)


    

    args = parser.parse_args()
    if args.totaltuningrate!=None:
        args.abtuningrate=args.totaltuningrate
        args.predtuningrate=args.totaltuningrate
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    num_files=len(datafiles)

    ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-3 for i in range(10,21,5)]
    #ebs=[1e-4,1e-3,1e-2]
    num_ebs=len(ebs)
    if args.blocksize>0:
        blocksize=args.blocksize
        algo="ALGO_INTERP_BLOCKED"
    else:
        blocksize=32 
        algo="ALGO_INTERP_LORENZO"


    if args.cr_tuning:
        
        tuning_target="TUNING_TARGET_CR"
    else:
        
        tuning_target="TUNING_TARGET_RD"
    cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    alpha=np.zeros((num_ebs,num_files),dtype=np.float32)
    beta=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    pid=os.getpid()
    
    configstr="[GlobalSettings]\nCmprAlgo = %s \ntuningTarget = %s \n[AlgoSettings]\nautoTuningRate = %f \npredictorTuningRate= %f \nlevelwisePredictionSelection = %d \nmaxStep = %d \ninterpolationBlockSize = %d \ntestLorenzo= %d \nlinearReduce= %d \n" % \
    (algo,tuning_target,args.abtuningrate,args.predtuningrate,args.levelwise,args.maxstep,blocksize,args.lorenzo,args.linear_reduce) 
    with open("%s.config" % pid,"w") as f:
        f.write(configstr)


    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            
            filepath=os.path.join(datafolder,datafile)

            
            comm="sz3_new -z -f -a -i %s -o %s.out -M REL %f -%d %s -c %s.config" % (filepath,pid,eb,args.dim," ".join(args.dims),pid)
            
            with os.popen(comm) as f:
                lines=f.read().splitlines()
                print(lines)
                
                r=eval(lines[-3].split('=')[-1])
                p=eval(lines[-6].split(',')[0].split('=')[-1])
                n=eval(lines[-6].split(',')[1].split('=')[-1])
                cr[i][j]=r 
                psnr[i][j]=p
                overall_psnr[i]+=n**2
                for line in lines:
                    if "alpha" in line:
                        #print(line)
                        a=eval(line.split(" ")[4][:-1])
                        b=eval(line.split(" ")[7][:-1])
                        alpha[i][j]=a
                        beta[i][j]=b
            
                
                

            
            comm="rm -f %s.out" % pid
            os.system(comm)
    comm="rm -f %s.config" % pid
    os.system(comm)
    overall_psnr=overall_psnr/num_files
    overall_psnr=np.sqrt(overall_psnr)
    overall_psnr=-20*np.log10(overall_psnr)
    overall_cr=np.reciprocal(np.mean(np.reciprocal(cr),axis=1))
            

    cr_df=pd.DataFrame(cr,index=ebs,columns=datafiles)
    psnr_df=pd.DataFrame(psnr,index=ebs,columns=datafiles)
    overall_cr_df=pd.DataFrame(overall_cr,index=ebs,columns=["overall_cr"])
    overall_psnr_df=pd.DataFrame(overall_psnr,index=ebs,columns=["overall_psnr"])
    alpha_df=pd.DataFrame(alpha,index=ebs,columns=datafiles)
    beta_df=pd.DataFrame(beta,index=ebs,columns=datafiles)
    cr_df.to_csv("%s_cr.tsv" % args.output,sep='\t')
    psnr_df.to_csv("%s_psnr.tsv" % args.output,sep='\t')
    overall_cr_df.to_csv("%s_overall_cr.tsv" % args.output,sep='\t')
    overall_psnr_df.to_csv("%s_overall_psnr.tsv" % args.output,sep='\t')
    alpha_df.to_csv("%s_alpha.tsv" % args.output,sep='\t')
    beta_df.to_csv("%s_beta.tsv" % args.output,sep='\t')