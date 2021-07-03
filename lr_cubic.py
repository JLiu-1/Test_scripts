import numpy as np 
from sklearn.linear_model import LinearRegression
from nyx_cubic import NYX_cubic
path="/home/jliu447/lossycompression/NYX"
field="baryon_density"
ratio=3

dataset=NYX_cubic(path,field,0,3,ratio=ratio,log=1)
print("finished reading data")
x=dataset.blocks
y=dataset.regs.flatten()
print("start regression")
reg=LinearRegression(fit_intercept=False).fit(x, y)

print(reg.coef_)
#a=list(reg.coef_).append(0)
#a=np.array(a,dtype=np.float32)
reg.coef_.astype(np.float32).tofile("lr_cubic_b.dat")
