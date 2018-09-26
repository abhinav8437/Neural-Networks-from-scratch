import numpy as np
from sklearn.preprocessing import scale,normalize


def predict_softmax_output(z):
    softmax  = np.exp(z) / np.sum(np.exp(z),axis=1).reshape(-1,1)
    return softmax

def derivative_of_softmax_output(z):
    return (1-predict_softmax_output(z))