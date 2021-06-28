import numpy as np 
from sklearn.linear_model import LinearRegression
from nyx import NYX
path="/home/jliu447/lossycompression/NYX"
field="baryon_density"
ratio=1

dataset=NYX(path,field,0,3,ratio=ratio,log=1)
print("finished reading data")
x=dataset.blocks
y=dataset.regs.flatten()
print("start regression")
reg=LinearRegression(fit_intercept=False).fit(x, y)

print(reg.coef_)
reg.coef_.tofile("lr_b.dat")
