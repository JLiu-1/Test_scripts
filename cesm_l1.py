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
    return np.sum(np.abs(y - fit(X, params)))


dataset=CESM(path,field,start=0,end=50,level=level,ratio=ratio)
print("finished reading data")
x=dataset.blocks.astype(np.double)
y=dataset.regs.astype(np.double).flatten()
print("start regression")
output = minimize(cost_function, x0=np.ones(x.shape[-1]), args=(x, y))
print(output)
np.array(output.x).tofile(outfile)
