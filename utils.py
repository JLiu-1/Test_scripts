import numpy as np 
#from sklearn.linear_model import LinearRegression
import math
#import random
from collections import Counter
def quantize(data,pred,error_bound):
    radius=32768
    
    diff = data - pred
    quant_index = (int) (abs(diff)/ error_bound) + 1
    #print(quant_index)
    if (quant_index < radius * 2) :
        quant_index =quant_index>> 1
        half_index = quant_index
        quant_index =quant_index<< 1
        #print(quant_index)
        quant_index_shifted=0
        if (diff < 0) :
            quant_index = -quant_index
            quant_index_shifted = radius - half_index
        else :
            quant_index_shifted = radius + half_index
        
        decompressed_data = pred + quant_index * error_bound
        #print(decompressed_data)
        if abs(decompressed_data - data) > error_bound :
            #print("b")
            return 0,data
        else:
            #print("c")
            data = decompressed_data
            return quant_index_shifted,data
        
    else:
        #print("a")
        return 0,data

def estimate_bitrate(quant_bins):
    bins=Counter()
    count=len(quant_bins)
    for b in quant_bins:
        bins[b]+=1

    bitrate=0
    for b in bins:
        p=bins[b]/count
        bitrate-=p*math.log(p,2)
    return bitrate

def interp_linear(x,y):#-1,1
    return (x+y)*0.5

def exterp_linear(x,y):#-3 -1
    return -0.5*x+1.5*y 

def interp_quad(a,b,c):#-1,1,3
    return (3*a+6*b-c)*0.125

def interp_quad2(a,b,c):#-3,-1,1
    return (-a+6*b+3*c)*0.125

def exterp_quad(a,b,c):#-5,-3,-1
    return (3 * a - 10 * b + 15 * c) *0.125

def interp_cubic(a,b,c,d):#-3,-1,1,3
    return (-a+9*b+9*c-d)*0.0625

def lor_2d(a,b,c):
    return b+c-a

def interp_2d(a,b,c,d):
    return (a+b+c+d)*0.25

def interp_3d(a,b,c,d,e,f):
    return (a+b+c+d+e+f)/6
