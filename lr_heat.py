import numpy as np 
from sklearn.linear_model import LinearRegression
from heat import Heat
path="/home/jliu447/lossycompression/Heat"

ratio=100

dataset=Heat(path,20000,20500,200,200,ratio=ratio)
print("finished reading data")
x=dataset.blocks.astype(np.float64)
y=dataset.regs.flatten().astype(np.float64)
print("start regression")
reg=LinearRegression(fit_intercept=True).fit(x, y)

print(reg.coef_)
print(reg.intercept_)
a=reg.coef_
a=np.concatenate( (a, reg.intercept_) ) 
a.tofile("lr_heat.dat")
