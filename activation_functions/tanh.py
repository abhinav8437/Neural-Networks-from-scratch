import numpy as np

def predict_tanh_output(z):
    return 2/(1+np.exp((-2)*z)) - 1

def derivative_tanh_output(z):
    derivative = 1 - np.power(z,2)
    return derivative