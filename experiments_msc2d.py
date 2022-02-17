import os 
import numpy as np

import argparse
import math
parser = argparse.ArgumentParser()

parser.add_argument('--input','-i',type=str)
parser.add_argument('--output','-o',type=str)
parser.add_argument('--max_step','-s',type=int,default=-1)
parser.add_argument('--min_coeff_level','-cl',type=int,default=99)
parser.add_argument('--rate','-r',type=float,default=1.0)
parser.add_argument('--maximum_rate','-m',type=float,default=10.0)
parser.add_argument('--anchor_rate','-a',type=float,default=0.0)
parser.add_argument('--multidim_level','-d',type=int,default=99)
parser.add_argument('--sz_interp','-n',type=int,default=0)
parser.add_argument('--rlist',type=float,default=-1,nargs="+")
parser.add_argument('--size_x','-x',type=int,default=1800)
parser.add_argument('--size_y','-y',type=int,default=3600)
parser.add_argument('--fix','-f',type=str,default="none")
parser.add_argument('--blockwise','-b',type=int,default=0)

args = parser.parse_args()
pid=str(os.getpid()).strip()
dout="%s_d.dat" %pid 
qout="%s_q.dat" %pid 
uout="%s_u.dat" % pid
ebs=[i*1e-3 for i in range(1,11)]+[1.5e-2,2e-2]
#ebs=[1e-3,1e-2]
if args.rlist!=-1:
    if isinstance(args.rlist,float):
        args.rlist=[args.rlist]

    rlist=" ".join([str(x) for x in args.rlist])
else:
    rlist="-1"
data=np.zeros((len(ebs)+1,2,2),dtype=np.float32)
for i in range(2):
    data[1:,0,i]=ebs
    #data[0,1:,i]=idxrange
if args.blockwise==1:
    script_name="multilevel_selective_compress_blockwise2d.py"
elif args.blockwise==2:
    script_name="multilevel_selective_compress_blockwise2d_rebuild.py"
else:
    script_name="multilevel_selective_compress_2d_api.py"
for i,eb in enumerate(ebs):
	command1="python %s -i %s -o %s -q %s -u %s -s %d -r %f -m %f -x %d -y %d -e %f -cl %d -a %f -d %d -n %d --rlist %s -f %s"\
	% (script_name,args.input, dout,qout,uout,args.max_step,args.rate,args.maximum_rate,args.size_x,args.size_y,eb,args.min_coeff_level,args.anchor_rate,args.multidim_level,args.sz_interp,rlist,args.fix)
	os.system(command1)
	command2="sz_backend %s %s " % (qout,uout)
	with os.popen(command2) as f:
		lines=f.read().splitlines()
		cr=eval(lines[4].split("=")[-1])
		if args.anchor_rate==0:
			anchor_ratio=1/(args.max_step**2)
            cr=1/((1-anchor_ratio)/cr+anchor_ratio/2)
        if args.blockwise>0:
            cr=1/(1/cr+3*math.log(args.max_step,2)/( 2*32*(args.max_step**2)) )
	command3="compareData -f %s %s" % (args.input,dout)
	with os.popen(command3) as f:
		lines=f.read().splitlines()
		psnr=eval(lines[6].split(',')[0].split('=')[1])
    
	data[i+1][1][0]=cr
	data[i+1][1][1]=psnr
	command4="rm -f %s;rm -f %s;rm -f %s" % (dout,qout,uout)

	os.system(command4)


np.savetxt("%s_final_cr.txt" % args.output,data[:,:,0],delimiter='\t')
np.savetxt("%s_psnr.txt" % args.output,data[:,:,1],delimiter='\t')
