import numpy as np
import pywt
import os
import sys
input_file=sys.argv[1]
output_folder=sys.argv[2]
eb=eval(sys.argv[3])
dims=[]
for i in range(4,len(sys.argv)):
 dims.append(eval(sys.argv[i]))
dims=tuple(dims)


if os.path.isfile(input_file):
    ifiles=[input_file]
else:
    ifiles= [os.path.join(input_file,file) for file in os.listdir(input_file) if file.split(".")[-1]=="dat" or file.split(".")[-1]=="f32" or file.split(".")[-1]=="bin"]

pid=os.getpid()

for ifile in ifiles:
    ifilename=os.path.basename(ifile)
    ofile=os.path.join(output_folder,ifilename+".tsv")
    fl=open(ofile,"w")
    fl.write("Wave\tCR\tPSNR\n")
    #dwt_waves=pywt.wavelist(kind='discrete')
    dwt_waves=["sym13","sym16","sym18"]
    a=np.fromfile(input_file,dtype=np.float32)
    a=a.reshape(dims)
    rng=np.max(a)-np.min(a)
    mean=np.mean(a)
    a=a-mean
    orisize=1
    for d in dims:
        orisize*=d
    for dwt in dwt_waves:
        print("processing %s"% dwt)
        try:
            c=pywt.wavedecn(a,dwt,mode="periodization")
            d=pywt.coeffs_to_array(c)
            d[0].tofile("%s_decn.test"%pid)
            s=reversed(list(d[0].shape))
            totalsize=1
            for x in d[0].shape:
                totalsize*=x
            s=[str(x) for x in s]
            shapestr=" ".join(s)
            print(shapestr)
      #do qoz
            command="qoz -z -f -a -i %s_decn.test -o %s_decn.test.out -%d %s -c szw_wt.config -M ABS %f" % (pid,pid,len(dims),shapestr,rng*eb)

      #extract cr psnr
            with os.popen(command) as p:
                lines=p.read().splitlines()
                cr=eval(lines[-3].split("=")[-1])
                cr*=float(orisize)/totalsize
       #psnr=eval(lines[-6].split(',')[0].split('=')[-1])
            nd=np.fromfile("%s_decn.test.out" % pid,dtype=np.float32).reshape(d[0].shape)
            c=pywt.array_to_coeffs(nd,d[1])
            b=pywt.waverecn(c,dwt,mode="periodization")
            if b.shape!=dims:
                if (len(dims)==2):
                    b=b[:dims[0],:dims[1]]
                else:
                    b=b[:dims[0],:dims[1],:dims[2]]
            b=b+mean
            b.tofile("%s_recn.test"%pid)
            with os.popen("compareData -f %s %s_recn.test" % (input_file,pid)) as p:
                lines=p.read().splitlines()

                psnr=eval(lines[-3].split(',')[0].split('=')[-1])
            print(cr)
            print(psnr)

            fl.write(dwt+"\t"+str(cr)+"\t"+str(psnr)+"\n")
        except:
            print("%s error" %dwt)

        os.system("rm -f *%s*"%pid)

    fl.close()