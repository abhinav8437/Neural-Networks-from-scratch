import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize,scale
import time
import paths
from activation_functions import sigmoid,softmax,tanh,relu


train_file = open(paths.train_set)
train_data = train_file.readlines()
test_file = open(paths.test_set)
test_data = test_file.readlines()

class neural_net_with_1_hidden_layer:
    def __init__(self,file,activation_function,loss_function,num_of_units,momentum):
        self.file = file
        self.costt = None
        self.weights1 = None
        self.weights2 = None
        self.activation_function = activation_function
        self.num_of_units = num_of_units
        self.loss_function = loss_function
        self.bias = momentum
        self.vdw_output = 0
        self.vdw_hidden = 0

    def load_data(self, file):
        X = []
        Y = []
        for line in range(len(file)):
            # split_line = file[line].split(',')
            # label = split_line[-1]
            label = file[line].split(",")[0]
            # x = [float(i) for i in split_line[:-1]]
            x = file[line].split(",")[1:]
            x = [int(i) for i in x]
            # Y.append(label[0])
            Y.append(label)
            X.append(x)
        return np.array(X)/255., Y


    def predict_layer(self, layer, weights):
        predicted_Y_of_all_data = np.dot(layer, weights)
        if (self.activation_function=="sigmoid"):
            return sigmoid.predict_sigmoid_output(predicted_Y_of_all_data)
        if (self.activation_function=="tanh"):
            return tanh.predict_tanh_output(predicted_Y_of_all_data)
        if (self.activation_function=="relu"):
            return relu.predicted_relu_output(predicted_Y_of_all_data)

    def predict_output(self, layer, weights):
        predicted_Y_of_all_data = np.dot(layer, weights)
        softmax_output = softmax.predict_softmax_output(predicted_Y_of_all_data)
        return softmax_output

    def cost_func(self, predicted_Y_of_all_data, output_Y_of_all_data):
        square_diff = np.power((predicted_Y_of_all_data - output_Y_of_all_data), 2)
        log = (-1)*np.multiply(output_Y_of_all_data.todense(),np.log(predicted_Y_of_all_data))
        # return np.sum(square_diff) / output_Y_of_all_data.shape[0]
        return np.sum(log)/predicted_Y_of_all_data.shape[0]

    def gradient_descent(self, dz2, predicted_hidden_layer,weights,batch_size,lambdaa,vdw):
        #adding Regularization term
        grad = (-1)*(np.dot(predicted_hidden_layer.T,dz2)) + lambdaa*weights/batch_size
        vdw1 = self.bias*vdw + (1-self.bias)*grad
        self.vdw_output = vdw1
        return grad
#
    def gradient_descent1(self, dz2, dz1, X, weights2,weights,batch_size,lambdaa,vdw):
        #adding regularization term
        grad = (-1)*np.dot(X.T, np.multiply((np.dot(dz2, weights2)), dz1)) + lambdaa*weights/batch_size
        vdw1 = self.bias * vdw + (1 - self.bias) * grad
        self.vdw_hidden = vdw1
        return grad

    def get_batches(self, inputs):
        return np.array(pd.DataFrame(inputs).sample(500))

    def fit(self, epochs,batch_size,lambdaa,lr):
        X_train, Y = self.load_data(self.file)
        weights1 = np.random.randn(X_train.shape[1],self.num_of_units)*np.sqrt(1/X_train.shape[1])
        weights2 = np.random.randn(self.num_of_units,len(set(Y)))*np.sqrt(1/self.num_of_units)
        one_hot = OneHotEncoder()
        output_Y_of_all_data = one_hot.fit_transform(np.array(Y).reshape(len(Y), 1))
        self.costt = []
        a = np.arange(0,X_train.shape[0]+1,batch_size) #this used in mini batch gradient
        for i in range(epochs):

            for batch in range(int(X_train.shape[0]/batch_size)):
                X_train_batch = X_train[a[batch]:a[batch + 1] - 1]
                output_Y_of_all_data_batch = output_Y_of_all_data[a[batch]:a[batch + 1] - 1]
                predicted_hidden_layer = self.predict_layer(X_train_batch, weights1)
                predicted_Y = self.predict_output(predicted_hidden_layer, weights2)
                cost = self.cost_func(predicted_Y, output_Y_of_all_data_batch)
                print (cost)
                z1 = np.dot(X_train_batch, weights1)
                z2 = np.dot(predicted_hidden_layer,weights2)

                if (self.loss_function=="mean_square_error"):
                    dz2 = np.multiply((output_Y_of_all_data_batch - predicted_Y),(1-predicted_Y))

                elif(self.loss_function=="cross_entropy"):
                    dz2 = np.multiply(output_Y_of_all_data_batch.todense(), softmax.derivative_of_softmax_output(z2))

                else:
                    print ("WRITE CORRECT LOSS FUNCTION")

                if (self.activation_function == "sigmoid"):
                    dz1 = sigmoid.derivative_of_sigmoid(z1)
                if (self.activation_function=='tanh'):
                    dz1 = tanh.derivative_tanh_output(z1)
                if (self.activation_function=='relu'):
                    dz1 = relu.derivative_relu_output(z1)

                # alpha = (1/(1+1*epochs))*lr
                alpha = lr
                #WEIGHTS UPDATED USING MINI BATCH GRADIENT DESCENT WITH MOMENTUM - 0.9
                weights2 = weights2 - alpha * (self.gradient_descent(dz2,predicted_hidden_layer,weights2,batch_size,lambdaa,self.vdw_output))
                weights1 = weights1 - alpha * (self.gradient_descent1(dz2,dz1,X_train_batch,weights2.T,weights1,batch_size,lambdaa,self.vdw_hidden))
                # print (weights2)

            self.costt.append(cost)
            self.weights2 = weights2
            self.weights1 = weights1

    def predict(self, file):
        X, Y = self.load_data(file)
        one_hot = OneHotEncoder()
        Y = one_hot.fit_transform(np.array(Y).reshape(len(Y), 1))
        predicted_hidden_layer = self.predict_layer(X, self.weights1)
        predict_output_layer = self.predict_output(predicted_hidden_layer, self.weights2)
        predicted_Y = np.argmax(predict_output_layer, axis=1)
        predicted_Y = [int(i) for i in predicted_Y]
        return predicted_Y


obj = neural_net_with_1_hidden_layer(train_data,activation_function = "tanh",loss_function="mean_square_error",num_of_units = 100,momentum=0.9)
obj.fit(epochs=6,batch_size = 500,lambdaa=0.2,lr = 0.0001)
#lambdaa is hyperparamter of regularization term added
#TEST_SET
X, Y = obj.load_data(test_data)
one_hot = OneHotEncoder()
output_Y_of_all_data = one_hot.fit_transform(np.array(Y).reshape(len(Y),1))
costt = obj.costt
actual_Y = [int(i) for i in Y]
predicted_Y = obj.predict(test_data)
print (accuracy_score(actual_Y,predicted_Y))