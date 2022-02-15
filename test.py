import numpy as np 

file="../JinyangLiu/SDRBENCH-CESM-ATM-cleared-1800x3600/CLDHGH_1_1800_3600.dat"

a=np.fromfile(file,dtype=np.float32).reshape((1800,3600))

max_step=64
lastx=(1799//max_step)*max_step
lasty=(3599//max_step)*max_step
ave_error=0
count=0
for x in range(2,lastx+1):
	for y in range(2,lasty+1):
		if x%2==0 and y%2==0:
			continue
		elif x%2==0:
			ave_error+=(a[x][y-2]+a[x][y+2])/2
		elif y%2==0:
			ave_error+=(a[x-2][y]+a[x+2][y])/2
		else:
			ave_error+=(a[x-2][y]+a[x+2][y]+a[x][y-2]+a[x][y+2])/4
		count+=1

b=ave_error/count
k=2
rng=np.max(a)-np.min(a)
e=1e-2*rng
print(0.2*b*k/e)