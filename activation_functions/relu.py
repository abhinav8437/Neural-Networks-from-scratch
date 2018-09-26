import numpy as np
from sklearn.preprocessing import normalize

def predicted_relu_output(z):
    # z = normalize(z)
    z[z<0]=0
    # print (z)
    return z

def derivative_relu_output(z):
    z[z>=0] = 1
    z[z<0] = 0
    return z