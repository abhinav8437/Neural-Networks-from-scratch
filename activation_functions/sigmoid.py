import numpy as np
from sklearn.preprocessing import normalize


def predict_sigmoid_output(z):
    z[z>700] = 0
    z = normalize(z)
    a =  1 / (1 + (np.exp((-1) * z)))
    return a


def derivative_of_sigmoid(z):
    return np.multiply(z, (1 - z))