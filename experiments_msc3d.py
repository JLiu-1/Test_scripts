import os 
import numpy as np
import math
import argparse
import pandas as pd
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input','-i',type=str)
    parser.add_argument('--output','-o',type=str)
    parser.add_argument('--max_step','-s',type=int,default=0)
    #parser.add_argument('--min_coeff_level','-cl',type=int,default=99)
    parser.add_argument('--rate','-r',type=float,default=-1)
    parser.add_argument('--maximum_rate','-m',type=float,default=-1)
    #parser.add_argument('--anchor_rate','-a',type=float,default=0.0)
    parser.add_argument('--multidim_level','-d',type=int,default=-1)
    parser.add_argument('--sz_interp','-n',type=int,default=1)
    parser.add_argument('--rlist',type=float,default=-1,nargs="+")
    parser.add_argument('--size_x','-x',type=int,default=129)
    parser.add_argument('--size_y','-y',type=int,default=129)
    parser.add_argument('--size_z','-z',type=int,default=129)
    parser.add_argument('--fix','-f',type=str,default="none")

    #parser.add_argument('--fullbound','-u',type=int,default=0)
    parser.add_argument('--anchor_fix','-c',type=int,default=0)
    parser.add_argument('--autotuning','-t',type=float,default=0.0)

    parser.add_argument('--block_size','-b',type=int,default=0)
    parser.add_argument('--interp_block_size',type=int,default=0)#interp block size
    parser.add_argument('--blockwise_tuning','-w',type=int,default=0)
    parser.add_argument('--order',type=str,default="block")
    #parser.add_argument('--rebuild','-e',type=int,default=0)
    args = parser.parse_args()
    print(args)
    pid=str(os.getpid()).strip()
    dout="%s_d.dat" %pid 
    qout="%s_q.dat" %pid 
    uout="%s_u.dat" % pid
    
    ebs=[i*1e-4 for i in range(1,10)]+[i*1e-3 for i in range(1,11)]+[1.5e-2,2e-2]
    num_ebs=len(ebs)
    datafolder=args.input
    datafiles=os.listdir(datafolder)
    datafiles=[file for file in datafiles if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]
    num_files=len(datafiles)
    if args.rlist!=-1:
        if isinstance(args.rlist,float):
            args.rlist=[args.rlist]

        rlist=" ".join([str(x) for x in args.rlist])
    else:
        rlist="-1"
    #ebs=[1e-3,1e-2]
    data=np.zeros((len(ebs)+1,2,2),dtype=np.float32)
    cr=np.zeros((num_ebs,num_files),dtype=np.float32)
    psnr=np.zeros((num_ebs,num_files),dtype=np.float32)
    alpha=np.zeros((num_ebs,num_files),dtype=np.float32)
    beta=np.zeros((num_ebs,num_files),dtype=np.float32)
    overall_cr=np.zeros((num_ebs,1),dtype=np.float32)
    overall_psnr=np.zeros((num_ebs,1),dtype=np.float32)


    if args.block_size>0:
        script_name="multilevel_selective_compress_blockwise3d_rebuild.py"

    else:
        script_name="multilevel_selective_compress_3d_api_rebuild.py"

    
    for i,eb in enumerate(ebs):
    
        for j,datafile in enumerate(datafiles):
            filepath=os.path.join(datafolder,datafile)
            command1="python %s -i %s -o %s -q %s -u %s -s %d -r %f -m %f -x %d -y %d -z %d -e %f  -d %d -n %d --rlist %s -f %s -t %f --interp_block_size %d"\
            % (script_name,filepath, dout,qout,uout,args.max_step,args.rate,args.maximum_rate,args.size_x,args.size_y,args.size_z,eb,\
                args.multidim_level,args.sz_interp,rlist,args.fix,int(1.0/args.autotuning),args.interp_block_size)
            if args.block_size>0:
                command1+=" -b %d -w %d --order %s" % (args.block_size,args.blockwise_tuning,args.order)
            with os.popen(command1) as f:
                lines=f.read().splitlines()
                for line in lines:
                    if "alpha" in line:
                        #print(line)
                        a=eval(line.split(" ")[4][:-1])
                        b=eval(line.split(" ")[7][:-1])
                        alpha[i][j]=a
                        beta[i][j]=b
            command2="sz_backend %s %s " % (qout,uout)
            with os.popen(command2) as f:
                lines=f.read().splitlines()
                r=eval(lines[4].split("=")[-1])
                if args.anchor_fix:
                    ele_num=args.size_x*args.size_y*args.size_z
                    anchor_num=((args.size_x-1)//args.max_step+1)*((args.size_y-1)//args.max_step+1)*((args.size_z-1)//args.max_step+1)
                #anchor_ratio=1/(args.max_step**2)
                    r=ele_num/((ele_num-anchor_num)/r+anchor_num)
                if args.block_size>0:
                    r=1/(1/r+2*math.log(args.max_step,2)/( 32*(args.block_size**3)) )

            command3="compareData -f %s %s" % (filepath,dout)
            with os.popen(command3) as f:
                lines=f.read().splitlines()
                p=eval(lines[-3].split(',')[0].split('=')[1])
                n=eval(lines[-3].split(',')[1].split('=')[-1])
            cr[i][j]=r 
            psnr[i][j]=p
            overall_psnr[i]+=n**2

        
        
            command4="rm -f %s;rm -f %s;rm -f %s" % (dout,qout,uout)
            os.system(command4)
            print(datafile,eb,a,b,r,p)


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
