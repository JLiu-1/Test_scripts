import numpy as np 
from sklearn.linear_model import LinearRegression
from heat import Heat
path="/home/jliu447/lossycompression/Heat"

ratio=1

dataset=Heat(path,20000,21000,200,200,ratio=ratio)
print("finished reading data")
x=dataset.blocks
y=dataset.regs.flatten()
print("start regression")
reg=LinearRegression(fit_intercept=False).fit(x, y)

print(reg.coef_)
reg.coef_.tofile("lr_heat.dat")
