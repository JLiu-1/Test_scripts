import numpy as np 
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from nyx import NYX
import sys
from cesm import *
path="../cesm-multisnapshot-5fields"
field="CLDHGH"
if len(sys.argv)>=2:
    level=int(sys.argv[1])
else:
    level=2
if len(sys.argv)>=3:
    ratio=int(sys.argv[2])
else:
    ratio=1
if len(sys.argv)>=4:
    outfile=sys.argv[3]
else:
    outfile="lr_cldhgh.dat"

def fit(X, params):
    return X.dot(params)


def cost_function(params, X, y):
    return np.mean(np.abs(y - fit(X, params)))+np.mean(np.abs(y - fit(X, params))**2)

init_2=np.array([-1.,1,1])
init_3=np.array([-1,2,-1,2,-4,2,-1,2])
if level==2:
    init=init_2
else:
    init=np.ones(8)
dataset=CESM(path,field,start=0,end=50,level=level,ratio=ratio)
print("finished reading data")
x=dataset.blocks.astype(np.double)
y=dataset.regs.astype(np.double).flatten()
print("start regression")
output = minimize(cost_function, x0=init, args=(x, y))
print(output)
np.array(output.x).tofile(outfile)
