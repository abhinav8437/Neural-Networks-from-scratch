# Neural-Networks-from-scratch
Dependencies

Numpy

Pandas

Seaborn - plot cost function of every iteration

Sklearn - for OneHotEncoder and Accuracy score

neural_net_1_layer and neural_net_2_layer are working
In neural_net_2_layer
These are the following arguments you have to pass when making an object -
training Data, Activation function, Loss Function, Number of units, Bias(Momentum, usually it is kept 0.9)
In Number of units, pass a list of two elements defining number of units in each layer

To fit, these are the arguments -
epochs, Batch size, Learning Rate
Using learning rate decay to optimize gradient descent.

It is highly probable to get Nan values after some epochs, ie vanishing gradients.

currently I am working on Generalised Neural net,in this this code will work on N number of layers.
You have to give a list of number of layers, number of units in each layer and till now it is
initializing the weights, doing 1 forward pass but yet have to get the generalised way to calculate gradient descent(Back-Propagation)


On Cifar10 dataset - 
Highest accuracy achieved after tweaking some hyperparameters

90% accuracy with 2 HIDDEN LAYER

89% accuracy with 1 HIDDEN LAYER


On MNIST dataset -

Achieved-

83% accuracy with NO HIDDEN LAYER

91% accuracy with 1 HIDDEN LAYER

93% accuracy with 2 HIDDEN LAYER
