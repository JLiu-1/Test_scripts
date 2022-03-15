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
    parser.add_argument('--config','-c',type=str,default=None)
    parser.add_argument('--ssim',"-s",type=int,default=0)
    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)
    

    args = parser.parse_args()
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    num_files=len(datafiles)

    ebs=[1e-5,5e-5]+[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-3 for i in range(10,21,5)]
    #ebs=[1e-4,1e-3,1e-2]
    num_ebs=len(ebs)

    cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    #nrmse=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    ssim=np.zeros((num_ebs,num_files),dtype=np.float32)
    #algo=np.zeros((num_ebs,num_files),dtype=np.int32)
    overall_ssim=np.zeros((num_ebs,1),dtype=np.float32)
    pid=os.getpid()
    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            
            filepath=os.path.join(datafolder,datafile)

            
            comm="test_compress %s 0 %f 3 1 %d %s" % (filepath,eb,args.dim," ".join(reversed(args.dims)))
            os.system(comm)
            comm="test_decompress %s %s.mgard 0 %d %s" % (filepath,filepath,args.dim," ".join(reversed(args.dims)))

            
            
            with os.popen(comm) as f:
                lines=f.read().splitlines()
                r=eval(lines[-1].split('=')[-1])
                p=eval(lines[-2].split(',')[-1].split('=')[-1])
               
                cr[i][j]=r 
                psnr[i][j]=p
                #overall_psnr[i]+=n**2
               
            comm="compareData -f %s %s.mgard.out " % (filepath,filepath)
            with os.popen(comm) as f:
                lines=f.read().splitlines()
                
                n=eval(lines[-3].split(',')[-1].split('=')[-1])
                
                overall_psnr[i]+=n**2
              


            if args.ssim:

                comm="calculateSSIM -f %s %s.out %s" % (filepath,pid," ".join(args.dims) )
                try:
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        print(lines)
                        s=eval(lines[-1].split('=')[-1])
                        ssim[i][j]=max(s,0)
                except:
                    ssim[i][j]=0

            
                
                

            
            comm="rm -f %s.out" % pid
            os.system(comm)
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
  

    if (args.ssim):
        overall_ssim=np.mean(ssim,axis=1)
        ssim_df=pd.DataFrame(ssim,index=ebs,columns=datafiles)
        overall_ssim_df=pd.DataFrame(overall_ssim,index=ebs,columns=["overall_ssim"])
        ssim_df.to_csv("%s_ssim.tsv" % args.output,sep='\t')
        overall_ssim_df.to_csv("%s_overall_ssim.tsv" % args.output,sep='\t')