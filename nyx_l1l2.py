import numpy as np 
from sklearn.linear_model import LinearRegression,Lasso
from scipy.optimize import minimize
from nyx import NYX
import sys
path="/home/jliu447/lossycompression/NYX"
field="baryon_density"
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
    outfile="lr_b.dat"

def fit(X, params):
    return X.dot(params)


def cost_function(params, X, y):
    return np.mean(np.abs(y - fit(X, params)))+np.mean(np.abs(y - fit(X, params))**2)
dataset=NYX(path,field,0,3,level=level,ratio=ratio,log=1)
print("finished reading data")
x=dataset.blocks.astype(np.double)
y=dataset.regs.astype(np.double).flatten()
print("start regression")

init_2=np.array([1.,-1,-1,1,-1,1,1])
init_3=np.ones(x.shape[-1])
if level==2:
    init=init_2
else:
    init=init_3

output = minimize(cost_function, x0=init, args=(x, y))
print(output)
np.array(output.x).tofile(outfile)
