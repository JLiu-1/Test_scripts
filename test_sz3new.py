import os
import numpy as np 
import argparse
import pandas as pd
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--input','-i',type=str)
    parser.add_argument('--output','-o',type=str)
    
   
    
    parser.add_argument('--dim','-d',type=int,default=3)
    parser.add_argument('--lorenzo','-z',type=int,default=0)
    parser.add_argument('--dims','-m',type=str,nargs="+")
    parser.add_argument('--levelwise','-l',type=int,default=0)
    parser.add_argument('--maxstep','-s',type=int,default=0)
    parser.add_argument('--blocksize','-b',type=int,default=0)
    parser.add_argument('--sample_blocksize','-e',type=int,default=0)
    
    parser.add_argument('--abtuningrate',"-a",type=float,default=0.005)
    parser.add_argument('--predtuningrate',"-p",type=float,default=0.005)
    parser.add_argument('--totaltuningrate',"-t",type=float,default=None)
    parser.add_argument('--tuning_target',"-n",type=str,default="rd")
    parser.add_argument('--linear_reduce',"-r",type=int,default=0)
    parser.add_argument('--multidim',"-u",type=int,default=0)
    parser.add_argument('--profiling',type=int,default=0)
    parser.add_argument('--fixblock',"-f",type=int,default=0)
    parser.add_argument('--ssim',type=int,default=0)
    parser.add_argument('--autocorr',"-c",type=int,default=0)
    parser.add_argument('--alpha',type=float,default=-1)
    parser.add_argument('--beta',type=float,default=-1)
    parser.add_argument('--bsbs',type=int,default=0)
    parser.add_argument('--sbsbs',type=int,default=0)
    
    parser.add_argument('--abconf',type=int,default=0)
    parser.add_argument('--pda',type=float,default=1.5)
    parser.add_argument('--pdb',type=float,default=2)
    parser.add_argument('--pdreal',type=int,default=0)
    parser.add_argument('--lastpdt',type=int,default=0)
    parser.add_argument('--ablist',type=int,default=0)
    parser.add_argument('--cross',type=int,default=0)
    parser.add_argument('--wavelet',type=int,default=0)
    parser.add_argument('--wrc',type=float,default=1.0)
    parser.add_argument('--waveletautotuning',type=int,default=0)
    #parser.add_argument('--external_wave','-x',type=int,default=0)
    #parser.add_argument('--wave_type',"-w",type=str)
    parser.add_argument('--field',type=str,default=None)
    parser.add_argument('--var_first',type=int,default=0)
    parser.add_argument('--sperr',type=int,default=-1)
    parser.add_argument('--conditioning',type=int,default=0)


    #parser.add_argument('--size_x','-x',type=int,default=1800)
    #parser.add_argument('--size_y','-y',type=int,default=3600)
    #parser.add_argument('--size_z','-z',type=int,default=512)
    '''
    if args.sample_blocksize==-1:
        sample_blocksize=64 if args.dim==2 else 32
    else:
        sample_blocksize=args.sample_blocksize
    '''



    

    args = parser.parse_args()
    if args.tuning_target=="ssim":
        args.ssim=1
    if args.tuning_target=="ac":
        args.autocorr=1
    if args.totaltuningrate!=None:
        args.abtuningrate=args.totaltuningrate
        args.predtuningrate=args.totaltuningrate
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    if args.field!=None:
        datafiles=[file for file in datafiles if args.field in file]
    num_files=len(datafiles)

    ebs=[1e-5,5e-5]+[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,10)]+[i*1e-3 for i in range(10,21,5)]
    #ebs=[i*1e-3 for i in range(1,10)]+[i*1e-3 for i in range(10,21,5)]
    #ebs=[1e-4,1e-3,1e-2]
    num_ebs=len(ebs)
    if args.blocksize>0:
        blocksize=args.blocksize
        algo="ALGO_INTERP_BLOCKED"
    else:
        blocksize=32 
        algo="ALGO_INTERP_LORENZO"
    
    tuning_target_dict={"rd":"TUNING_TARGET_RD","cr":"TUNING_TARGET_CR","ssim":"TUNING_TARGET_SSIM","ac":"TUNING_TARGET_AC"}

    
        
    tuning_target=tuning_target_dict[args.tuning_target]
    
    cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    ssim=np.zeros((num_ebs,num_files),dtype=np.float32)
    alpha=np.zeros((num_ebs,num_files),dtype=np.float32)
    beta=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_ssim=np.zeros((num_ebs,1),dtype=np.float32)
    ac=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_ac=np.zeros((num_ebs,1),dtype=np.float32)
    wavelet_selection=np.zeros((num_ebs,num_files),dtype=np.int32)
    pid=os.getpid()
    
    configstr="[GlobalSettings]\nCmprAlgo = %s \ntuningTarget = %s \n[AlgoSettings]\nautoTuningRate = %f \npredictorTuningRate= %f \nlevelwisePredictionSelection = %d \nmaxStep =\
     %d \ninterpBlockSize = %d \ntestLorenzo = %d \nlinearReduce = %d \nmultiDimInterp = %d \nsampleBlockSize = %d \nprofiling = %d \nfixBlockSize = %d \nalpha = %f \nbeta = \
     %f \npdTuningAbConf = %d \npdAlpha = %d \npdBeta = %d \npdTuningRealComp = %d \nlastPdTuning = %d \nabList = %d \nblockwiseSampleBlockSize = %d \ncrossBlock = \
     %d \nsampleBlockSampleBlockSize = %d \nwavelet = %d\nwavelet_rel_coeff = %f\npid = %s\nwaveletAutoTuning = %d\nvar_first = %d\nsperr = %d\nconditioning = %d\n"\
     % (algo,tuning_target,args.abtuningrate,args.predtuningrate,args.levelwise,args.maxstep,blocksize,args.lorenzo,args.linear_reduce,args.multidim,args.sample_blocksize,\
        args.profiling,args.fixblock,args.alpha,args.beta,args.abconf,args.pda,args.pdb,args.pdreal,args.lastpdt,args.ablist,args.bsbs,args.cross,args.sbsbs,args.wavelet,\
        args.wrc,pid,args.waveletautotuning,args.var_first,args.sperr,args.conditioning) 
    with open("%s.config" % pid,"w") as f:
        f.write(configstr)
    for i,eb in enumerate(ebs):
        for j,datafile in enumerate(datafiles):
            filepath=os.path.join(datafolder,datafile)
            '''
            if args.external_wave:
                command="python coeff_dwt.py %s %s %s %s" % (filepath,args.wave_type,pid," ".join(args.dims))
                os.system(command)
            '''
            

            
            comm="qoz -z -f -a -i %s -o %s.out -M REL %f -%d %s -c %s.config" % (filepath,pid,eb,args.dim," ".join(args.dims),pid)
            
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
                    if "Selected wavelet" in line:
                        wv=eval(line.split(":")[-1])
                        wavelet_selection[i][j]=wv
            
                
            if args.ssim:
                comm="calculateSSIM -f %s %s.out %s" % (filepath,pid," ".join(args.dims))
                try:
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        print(lines)
                        s=eval(lines[-1].split('=')[-1])
                        ssim[i][j]=max(s,0)
                except:
                    ssim[i][j]=0

            if args.autocorr:

                comm="computeErrAutoCorrelation -f %s %s.out " % (filepath,pid)
                try:
                    with os.popen(comm) as f:
                        lines=f.read().splitlines()
                        print(lines)
                        a=eval(lines[-1].split(':')[-1])
                        ac[i][j]=a
                except:
                    ac[i][j]=1




            
            comm="rm -f %s.out" % pid
            os.system(comm)
    comm="rm -f *%s*" % pid
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

    if args.ssim: 
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
    if args.waveletautotuning:
        wvs_df=pd.DataFrame(wavelet_selection,index=ebs,columns=datafiles)
        wvs_df.to_csv("%s_wave_selection.tsv" % args.output,sep='\t')