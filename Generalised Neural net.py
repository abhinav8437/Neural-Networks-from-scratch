import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize,scale
import time
import paths
from activation_functions import sigmoid,softmax,tanh,relu
import time
import tensorflow as tf

train_file = open(paths.train_set)
train_data = train_file.readlines()
test_file = open(paths.test_set)
test_data = test_file.readlines()

class neural_net_with_1_hidden_layer:
    def __init__(self,file,activation_function,num_of_layers,loss_function,num_of_units,bias):
        self.file = file
        self.costt = None
        self.num_of_layers = num_of_layers
        self.weight_list = []
        self.activation_function = activation_function
        self.units_in_layer = num_of_units
        self.bias = bias
        self.loss_function = loss_function
        self.vdw_output = 0
        self.vdw_hidden = 0

    def load_data(self, file):
        X = []
        Y = []
        for line in range(len(file)):
            split_line = file[line].split(',')
            label = split_line[-1]
            dimension = [float(i) for i in split_line[:-1]]
            Y.append(label[0])
            X.append(dimension)
        return np.array(X), Y

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
        # square_diff = np.power((predicted_Y_of_all_data - output_Y_of_all_data), 2)
        log = (-1)*np.multiply(output_Y_of_all_data.todense(),np.log(predicted_Y_of_all_data))
        # return np.sum(square_diff) / output_Y_of_all_data.shape[0]
        return np.sum(log)/predicted_Y_of_all_data.shape[0]

    def gradient_descent(self, dz3, predicted_hidden_layer,vdw):
        grad = (-1)*(np.dot(predicted_hidden_layer.T,dz3))
        vdw1 = self.bias*vdw + (1-self.bias)*grad
        self.vdw_output = vdw1
        return grad

    def gradient_descent1(self, dz3, dz2,layer, weights3,vdw):
        grad = (-1)*np.dot(layer.T, np.multiply((np.dot(dz3, weights3)), dz2))
        vdw1 = self.bias * vdw + (1 - self.bias) * grad
        self.vdw_hidden = vdw1
        return grad

    def gradient_descent2(self,dz3,dz2,dz1,weights3,X_train,weights2):
        grad = (-1)*np.dot(X_train.T,np.multiply(np.dot(np.multiply((np.dot(dz3, weights3)), dz2),weights2),dz1))
        return grad

    def fit(self, epochs,batch_size):
        X_train, Y = self.load_data(self.file)
        for i in range(len(self.units_in_layer)+1):#REMEMBER IF 3 LAYERS ARE GIVEN 4 WIGHTS WILL BE INITIALIZED
            if (i==0):
                weights = np.random.randn(X_train.shape[1],self.units_in_layer[i])*np.sqrt(1/X_train.shape[1])
                self.weight_list.append(weights)
            elif(i==self.num_of_layers):
                weights = np.random.randn(self.units_in_layer[-1],len(set(Y)))
                self.weight_list.append(weights)
            else:
                weights = np.random.randn(self.units_in_layer[i-1],self.units_in_layer[i])*np.sqrt(1/self.units_in_layer[i])
                self.weight_list.append(weights)
        # print ("WEIGHTS==>",self.weight_list[0].shape,self.weight_list[1].shape,self.weight_list[2].shape)
        one_hot = OneHotEncoder()
        output_Y_of_all_data = one_hot.fit_transform(np.array(Y).reshape(len(Y), 1))
        a = np.arange(0,X_train.shape[0]+1,batch_size) #this used in mini batch gradient

        for i in range(epochs):

            for batch in range(int(X_train.shape[0]/batch_size)):
                X_train_batch = X_train[a[batch]:a[batch + 1] - 1]
                output_Y_of_all_data_batch = output_Y_of_all_data[a[batch]:a[batch + 1] - 1]
                predicted_hidden_layer_list = []
                for weight in range(len(self.units_in_layer)+1):
                    if (weight==0):
                        predicted_layer = self.predict_layer(X_train_batch,self.weight_list[weight])
                        predicted_hidden_layer_list.append(predicted_layer)

                    elif (weight==self.num_of_layers):
                        predicted_layer = self.predict_output(predicted_hidden_layer_list[-1],self.weight_list[weight])
                        predicted_hidden_layer_list.append(predicted_layer)
                    else:
                        predicted_layer = self.predict_layer(predicted_hidden_layer_list[weight-1],self.weight_list[weight])
                        predicted_hidden_layer_list.append(predicted_layer)
                # print("P_LAYER==>", predicted_hidden_layer_list[0].shape, predicted_hidden_layer_list[1].shape, predicted_hidden_layer_list[2].shape)

                predicted_Y = predicted_hidden_layer_list[-1]
                self.costt = []
                cost = self.cost_func(predicted_Y, output_Y_of_all_data_batch)
                # print (cost)
                z_list = []
                for layer in range(len(self.units_in_layer)+1):
                    if (layer==0):
                        z_layer = np.dot(X_train_batch,self.weight_list[layer])
                        z_list.append(z_layer)
                    else:
                        z_layer = np.dot(predicted_hidden_layer_list[layer-1],self.weight_list[layer])
                        z_list.append(z_layer)
                # print ("Z_Layer==>",z_list[0].shape,z_list[1].shape,z_list[2].shape)

                #LOSS FUNCTION
                if (self.loss_function=="mean_square_error"):
                    dz3 = np.multiply((output_Y_of_all_data_batch - predicted_Y),(1-predicted_Y))

                elif(self.loss_function=="cross_entropy"):
                    dz3 = np.multiply(output_Y_of_all_data_batch.todense(), softmax.derivative_of_softmax_output(z_list[-1]))

                else:
                    print ("WRITE CORRECT LOSS FUNCTION")

                z_list.pop(-1)

                #DERIVATIVE OF HIDDEN LAYERS
                if (self.activation_function == "sigmoid"):
                    dz_list = [sigmoid.derivative_of_sigmoid(i) for i in z_list]
                elif (self.activation_function=='tanh'):
                    dz_list = [tanh.derivative_tanh_output(i) for i in z_list]
                elif (self.activation_function=='relu'):
                    dz_list = [relu.derivative_relu_output(i) for i in z_list]
                print (dz_list[0].shape,dz_list[1].shape,dz_list[2].shape)

                # alpha = (1/(1+1*epochs))*0.001
                alpha = 0.00001
                for update_param in range(len(self.units_in_layer)+1):
                    if (update_param==0):
                        self.weight_list[-1] = self.weight_list[-1] - alpha * (self.gradient_descent(dz3,predicted_hidden_layer_list[-2],self.vdw_output))
                    # elif(update_param==)
                    elif(update_param==1):
                        self.weight_list[-2] = self.weight_list[-2] - alpha * (self.gradient_descent1(dz3,dz_list[-1],predicted_hidden_layer_list[-3],self.weight_list[-1].T,self.vdw_hidden))

                    elif(update_param==2):
                        self.weight_list[-3] = self.weight_list[-3] - alpha * (self.gradient_descent2(dz3,dz_list[-1],dz_list[-2],self.weight_list[-1].T,X_train_batch,self.weight_list[-2].T))

            self.costt.append(cost)

    def predict(self, file):
        X, Y = self.load_data(file)
        predicted_hidden_layer_1 = self.predict_layer(X, self.weight_list[0])
        predicted_hidden_layer_2 = self.predict_layer(predicted_hidden_layer_1,self.weight_list[1])
        predict_output_layer = self.predict_output(predicted_hidden_layer_2, self.weight_list[2])
        predicted_Y = np.argmax(predict_output_layer, axis=1)
        predicted_Y = [int(i) for i in predicted_Y]
        return predicted_Y




obj = neural_net_with_1_hidden_layer(train_data,activation_function = "tanh",num_of_layers = 3,loss_function="mean_square_error",num_of_units = [100,90,80],bias=0.9)
time1 = time.time()
obj.fit(epochs=40,batch_size = 500)
time2 = time.time()
print ("total time took in training=====>>",time2-time1)
#TEST_SET
X, Y = obj.load_data(test_data)
one_hot = OneHotEncoder()
output_Y_of_all_data = one_hot.fit_transform(np.array(Y).reshape(len(Y),1))
costt = obj.costt
actual_Y = [int(i) for i in Y]
predicted_Y = obj.predict(test_data)
print (set(predicted_Y))
print (accuracy_score(actual_Y,predicted_Y))