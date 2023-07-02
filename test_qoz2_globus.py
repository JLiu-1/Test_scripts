import os
import numpy as np 
import argparse
import pandas as pd
import shutil
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--input','-i',type=str)
    parser.add_argument('--output','-o',type=str)
    parser.add_argument('--tsvout','-t',type=str)
    parser.add_argument('--error','-e',type=float)

    
   
    
    parser.add_argument('--dim','-d',type=int,default=2)
    parser.add_argument('--copy','-c',type=int,default=1)
    parser.add_argument('--level','-l',type=int,default=4)
    parser.add_argument('--dims','-m',type=str,nargs="+")
    parser.add_argument('--tuning_target',"-n",type=str,default="PSNR")
    #parser.add_argument('--config','-c',type=str,default=None)
    #parser.add_argument('--ssim',"-s",type=int,default=0)
    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)
    

    args = parser.parse_args()
    datafolder=args.input
    cmpfolder=args.output
    if not os.path.exists(cmpfolder):
        os.makedirs(cmpfolder)
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    num_files=len(datafiles)

    #ebs=[1e-4,1e-31]
    eb=args.error
    c_speed=np.zeros((1),dtype=np.float32)
    d_speed=np.zeros((1),dtype=np.float32)
    #nrmse=np.zeros((num_ebs,num_files),dtype=np.float32)
    #overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    #overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    #ssim=np.zeros((num_ebs,num_files),dtype=np.float32)
    #algo=np.zeros((num_ebs,num_files),dtype=np.int32)
    #overall_ssim=np.zeros((num_ebs,1),dtype=np.float32)
    pid=os.getpid()
    total_data_size=num_files
    for d in args.dims:
        total_data_size*=eval(d)
    total_data_size=total_data_size*4/(1024*1024)
    print(total_data_size)
    
    for j,datafile in enumerate(datafiles):
        filepath=os.path.join(datafolder,datafile)

        cmppath=os.path.join(cmpfolder,datafile)+".qoz2"
        comm="qoz -z -f -a -i %s -z %s -o %s.out -M REL -R %f -%d %s -T %s -q %d" % (filepath,cmppath,pid,eb,args.dim," ".join(args.dims),args.tuning_target,args.level)
        #print(comm)
        
        
        with os.popen(comm) as f:
            lines=f.read().splitlines()
            print(lines)
            ct=eval(lines[-12].split('=')[-1])
            dt=eval(lines[-2].split('=')[-1].split("s")[0])
            c_speed[0]+=ct
            d_speed[0]+=dt

            
            
       

        
            
            

        
        comm="rm -f %s.out" % pid
        os.system(comm)
    
    c_speed=total_data_size*np.reciprocal(c_speed)
    d_speed=total_data_size*np.reciprocal(d_speed)


   
    cs_df=pd.DataFrame(c_speed,index=[eb],columns=["Compression Speed (MB/s)"])
    ds_df=pd.DataFrame(d_speed,index=[eb],columns=["Decompression Speed (MB/s)"])
    
    
    cs_df.to_csv("%s_cspeed.tsv" % args.tsvout,sep='\t')
    ds_df.to_csv("%s_dspeed.tsv" % args.tsvout,sep='\t')

    if(args.copy>1):
        for i in range(1,args.copy):
            for j,datafile in enumerate(datafiles):
                cmppath=os.path.join(cmpfolder,datafile)+".qoz2"
                cmppath_i=cmppath+"."+str(i)
                shutil.copy(cmppath,cmppath_i)


   

    