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
    #parser.add_argument('--large_eb','-l',type=int,default=0)
    parser.add_argument('--ssim',"-s",type=int,default=0)
    parser.add_argument('--autocorr',"-a",type=int,default=0)
    #parser.add_argument('--config','-c',type=str,default=None)
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
    #maxpwerr=np.zeros((len(ebs)+1,len(idxlist)+1),dtype=np.float32)
    #nrmse=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    ssim=np.zeros((num_ebs,num_files),dtype=np.float32)
   
    overall_ssim=np.zeros((num_ebs,1),dtype=np.float32)
    ac=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_ac=np.zeros((num_ebs,1),dtype=np.float32)

    c_speed=np.zeros((num_ebs),dtype=np.float32)
    d_speed=np.zeros((num_ebs),dtype=np.float32)
    total_data_size=num_files
    for d in args.dims:
        total_data_size*=eval(d)
    total_data_size=total_data_size*4/(1024*1024)
    #algo=np.zeros((num_ebs,num_files),dtype=np.int32)
    pid=os.getpid()
    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            
            filepath=os.path.join(datafolder,datafile)
           
            
            arr=np.fromfile(filepath,dtype=np.float32)
            rng=np.max(arr)-np.min(arr)
            abseb=rng*eb
            if args.dim==2 and abseb<1e-8:
                abseb=1e-8
            if args.dim==2:
                comm="compressor_2d --show_stats --dims %s -o %s.sperr --pwe %f %s &>%s.txt;decompressor_2d -o %s.sperr.out %s.sperr >>%s.txt;compareData -f %s %s.sperr.out>>%s.txt" % (" ".join(args.dims),pid,abseb,filepath,pid,pid,pid,pid,filepath,pid,pid)
            else:
                comm="compressor_3d --show_stats --dims %s -o %s.sperr --pwe %f %s &>%s.txt;decompressor_3d -o %s.sperr.out %s.sperr >>%s.txt;compareData -f %s %s.sperr.out>>%s.txt" % (" ".join(args.dims),pid,abseb,filepath,pid,pid,pid,pid,filepath,pid,pid)
            #print(comm)
            os.system(comm)
            with open("%s.txt"%pid,"r") as f:
                lines=f.read().splitlines()
                try:
                    p=eval(lines[-3].split(',')[0].split('=')[-1])
                    n=eval(lines[-3].split(',')[1].split('=')[-1])
                except:
                    p=np.inf
                    n=np.inf
                #print(lines)
                for line in lines:
                    if "Average bpp" in line:
                        r=32/float(line.split(',')[0].split('=')[-1])
                    elif "Compression time" in line:

                        ct=eval(line.split('=')[-1].split('s')[0])
                    elif "Decompression time" in line:
                        dt=eval(line.split('=')[-1].split('s')[0])


                c_speed[i]+=ct
                d_speed[i]+=dt
                cr[i][j]=r 
                psnr[i][j]=p
                overall_psnr[i]+=n**2

            if args.ssim:

                comm="calculateSSIM -f %s %s.sperr.out %s" % (filepath,pid," ".join(args.dims))
                try:
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        print(lines)
                        s=eval(lines[-1].split('=')[-1])
                        ssim[i][j]=max(s,0)
                except:
                    ssim[i][j]=0

            if args.autocorr:

                comm="computeErrAutoCorrelation -f %s %s.sperr.out " % (filepath,pid)
                try:
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        print(lines)
                        a=eval(lines[-1].split(':')[-1])
                        ac[i][j]=a
                except:
                    ac[i][j]=1

            os.system("rm -f %s*" % pid)
            print(datafile,eb,rng,abseb,r,p)

    overall_psnr=overall_psnr/num_files
    overall_psnr=np.sqrt(overall_psnr)
    overall_psnr=-20*np.log10(overall_psnr)
    overall_cr=np.reciprocal(np.mean(np.reciprocal(cr),axis=1))
    c_speed=total_data_size*np.reciprocal(c_speed)
    d_speed=total_data_size*np.reciprocal(d_speed)



    cr_df=pd.DataFrame(cr,index=ebs,columns=datafiles)
    psnr_df=pd.DataFrame(psnr,index=ebs,columns=datafiles)
    overall_cr_df=pd.DataFrame(overall_cr,index=ebs,columns=["overall_cr"])
    overall_psnr_df=pd.DataFrame(overall_psnr,index=ebs,columns=["overall_psnr"])
    cs_df=pd.DataFrame(c_speed,index=ebs,columns=["Compression Speed (MB/s)"])
    ds_df=pd.DataFrame(d_speed,index=ebs,columns=["Decompression Speed (MB/s)"])
    
    cr_df.to_csv("%s_cr.tsv" % args.output,sep='\t')
    psnr_df.to_csv("%s_psnr.tsv" % args.output,sep='\t')
    overall_cr_df.to_csv("%s_overall_cr.tsv" % args.output,sep='\t')
    overall_psnr_df.to_csv("%s_overall_psnr.tsv" % args.output,sep='\t')
    cs_df.to_csv("%s_cspeed.tsv" % args.output,sep='\t')
    ds_df.to_csv("%s_dspeed.tsv" % args.output,sep='\t')
    if (args.ssim):
        overall_ssim=np.mean(ssim,axis=1)
        ssim_df=pd.DataFrame(ssim,index=ebs,columns=datafiles)
        overall_ssim_df=pd.DataFrame(overall_ssim,index=ebs,columns=["overall_ssim"])
        ssim_df.to_csv("%s_ssim.tsv" % args.output,sep='\t')
        overall_ssim_df.to_csv("%s_overall_ssim.tsv" % args.output,sep='\t')
        
    if (args.autocorr):
        overall_ac=np.mean(ac,axis=1)
        ac_df=pd.DataFrame(ac,index=ebs,columns=datafiles)
        overall_ac_df=pd.DataFrame(overall_ac,index=ebs,columns=["overall_ac"])
        ac_df.to_csv("%s_ac.tsv" % args.output,sep='\t')
        overall_ac_df.to_csv("%s_overall_ac.tsv" % args.output,sep='\t')
   
               
        
     

        
   


