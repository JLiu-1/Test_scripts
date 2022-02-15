import numpy as np 
import sys
#reb=float(sys.argv[1])
file="../JinyangLiu/SDRBENCH-CESM-ATM-cleared-1800x3600/CLDHGH_1_1800_3600.dat"

a=np.fromfile(file,dtype=np.float32).reshape((1800,3600))

max_step=64
lastx=(1799//max_step)*max_step
lasty=(3599//max_step)*max_step
ave_error=0
count=0
for x in range(3,lastx+1):
	for y in range(3,lasty+1):
		if x%2==0 and y%2==0:
			continue
		elif x%2==0:
			ave_error+=(-a[x][y-3]+9*a[x][y-1]+9*a[x][y+1]-a[x][y+3])/16
		elif y%2==0:
			ave_error+=(-a[x-3][y]+9*a[x-1][y]+9*a[x+1][y]-a[x+3][y])/16
		else:
			ave_error+=(-a[x][y-3]+9*a[x][y-1]+9*a[x][y+1]-a[x][y+3])/32+(-a[x-3][y]+9*a[x-1][y]+9*a[x+1][y]-a[x+3][y])/32
		count+=1

b=ave_error/count
k=8/9
rng=np.max(a)-np.min(a)
e=rng
print(0.2*b*k/e)