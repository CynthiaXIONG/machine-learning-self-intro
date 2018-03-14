#code style: https://google.github.io/styleguide/pyguide.html
import os
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import h5py

import tensorflow as tf
from tensorflow.python.framework import ops
import datasets.py_datasets as py_datasets

# DATA
def load_train_test_dataset(train_file_path, test_file_path):
    train_dataset = h5py.File(train_file_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_file_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def shuffle_data_set(X, Y, seed=None):
    if (seed is not None):
        np.random.seed(seed)            # To make the "random shuffle" the same, for testing purposes

    m = Y.shape[1]
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
    return shuffled_X, shuffled_Y

def split_data_set(data, buckets=[.6,.2,.2]):
    assert (np.sum(buckets) == 1.0)
    m = data.shape[1]
    splitted_data = []
    current_split = 0
    for i in range(len(buckets)):
        next_split = math.floor(min(buckets[i], 1.0) * m)
        splitted_data.append(data[:, current_split:next_split])

    return splitted_data

def split_data_set_to_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)
    """
 
    m = X.shape[1]
    mini_batches = []

    # Partition the mini-batches
    num_minibatches = math.ceil(m/mini_batch_size)
    for k in range(0, num_minibatches):
        mini_batch_X = X[:, mini_batch_size*k : min(mini_batch_size*(k+1), m)]
        mini_batch_Y = Y[:, mini_batch_size*k : min(mini_batch_size*(k+1), m)]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

#--------- DATA PROCESSING ---------#
def flatten_matrix(m):
    dim_size = 1
    for i in range(1, len(m.shape)):
        dim_size *= m.shape[i] 

    return m.reshape(-1, dim_size).T

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def normalization(x, feature_axis=1):
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord=None, axis=feature_axis, keepdims=True)
    # Divide x by its norm.
    return x / x_norm

def min_max_scalling(x, feature_axis=1):
    # Compute x_min and x_max
    x_min = np.min(x, axis=feature_axis, keepdims=True)
    x_max = np.max(x, axis=feature_axis, keepdims=True)
    # Scale
    return (x -x_min) / (x_max - x_min)

def feature_standardization(x, axis=1):
    # https://en.wikipedia.org/wiki/Feature_scaling
    # Subtract the mean (to have zero mean) and devide by the std
    return (x - np.mean(x, axis=axis, keepdims=True)) / np.std(x, axis=axis, keepdims=True)

#--------- VISUALIZATION ---------#
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)

#--------- ACTIVATION FUNCTIONS ---------#
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

def relu(Z):
    return np.maximum(0, Z)

def leaky_relu(Z):
    positive = np.multiply(Z > 0, Z)
    negative = p.multiply(Z <= 0, Z * 0.01)
    return positive + negative

    #--------- ACTIVATION FUNCTIONS DERIVATIVES ---------#
def sigmoid_derivative(dA, A):
    dZ = dA * A * (1-A)
    return dZ

def tanh_derivative(dA, Z):
    return 1 #1 - np.power(np.tanh(z), 2)  <- wrong?...needs dA?

def relu_derivative(dA, A):
    return np.multiply((A > 0), dA)

def leaky_relu_derivative(dA, A):
    positive = np.multiply((A > 0), dA)
    negative = np.multiply((A <= 0), 0.01 * dA)
    return positive + negative

#--------- PARAMETER INITIALIZATION ---------#
def initialize_parameters_with_zero(dim):
    w = np.zeros([dim, 1])
    b = 0
    return w, b

#--------- PERFORMANCE METRIC ---------#
def calc_accuracy(y_predict, y):
    return 1 - np.mean(np.abs(y_predict - y))

#--------- DEEP NET (L layers) ---------#
class DeepNN():
    def __init__(self, layers):
        self.layers = layers

    @staticmethod
    def initialize_parameters(X, layers):
        all_layers = layers.copy()
        all_layers.insert(0, {"n":X.shape[0]}) #add the inputs as the first layer

        parameters = {}
        L = len(all_layers)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(all_layers[l]["n"], all_layers[l-1]["n"]) * np.sqrt(2/all_layers[l-1]["n"])
            parameters['b' + str(l)] = np.zeros((all_layers[l]["n"], 1))

        return parameters

    @staticmethod
    def initialize_adam(parameters):
        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}
        
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
            s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
            s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
        adam_parameters = {}
        adam_parameters["v"] = v
        adam_parameters["s"] = s

        adam_parameters["t"] = 0    #Adam update counter
        
        return adam_parameters

    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation, dropout_keep_prob):
        linear_cache = (A_prev, W, b, dropout_keep_prob)
        Z = np.dot(W, A_prev) + b
        
        if activation == "sigmoid":
            A = sigmoid(Z) 
        elif activation == "tanh":
            A = tanh(Z)
        elif activation == "relu":
            A = relu(Z)
        elif activation == "leaky_relu":
            A = leaky_relu(Z)

        if (dropout_keep_prob < 1.0): #dropout regularization
            D = np.random.rand(A.shape[0], A.shape[1]) < dropout_keep_prob
            A = np.multiply(A, D)
            A /= dropout_keep_prob
        else:
            D = None
        
        activation_cache = (Z, D)
        cache = (linear_cache, activation_cache)
        return A, cache

    @staticmethod
    def forward_prop(X, parameters, layers):
        caches = []
        A = X
        L = len(layers)

        for l in range (L):
            A_prev = A
            A, cache = DeepNN.linear_activation_forward(A_prev, parameters['W' + str(l+1)], parameters['b' + str(l+1)], activation=layers[l]["activation"], dropout_keep_prob=layers[l]["dropout_keep_prob"])

            caches.append(cache)
        
        return A, caches

    @staticmethod
    def compute_cost(AL, Y, parameters=None, l2_reg_lambda=0.0, epsilon=1e-8):
        m = AL.shape[1]
        cost = -(1/m) * np.sum((np.dot(Y, np.log(AL + epsilon).T) + np.dot((1-Y), np.log(1-AL + epsilon).T)))
        
        #add L2 regularization
        if (l2_reg_lambda > 0.0):
            L = len(parameters) // 2
            weight_sum = 0
            for l in range(L):
                weight_sum += np.sum(np.square(parameters["W" + str(l+1)]))
            cost += (1/m) * (l2_reg_lambda/2) * weight_sum

        assert(cost.shape == ())
        return np.squeeze(cost)

    @staticmethod
    def linear_activation_backward(dA, A, cache, activation, l2_reg_lambda):
        #get cached values
        linear_cache, activation_cache = cache
        next_A, W, _, dropout_keep_prob = linear_cache
        Z, D = activation_cache

        if (D is not None):
            dA = np.multiply(dA, D)
            dA /= dropout_keep_prob

        #calc dZ from derivative of the activation function, dA
        if activation == "sigmoid":
            dZ = sigmoid_derivative(dA, A) 
        elif activation == "tanh":
            dZ = tanh_derivative(dA, Z) 
        elif activation == "relu":
            dZ = relu_derivative(dA, A) 
        elif activation == "leaky_relu":
            dZ = leaky_relu_derivative(dA, A) 

        m = dA.shape[1]
        dW = 1./m * np.dot(dZ, next_A.T) + (l2_reg_lambda/m)*W
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)

        return dZ, dA_prev, dW, db

    @staticmethod
    def backward_prop(X, Y, AL, caches, layers, l2_reg_lambda=0.0, epsilon = 1e-8):
        L = len(layers)
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        grads = {}

        # Initializing the backpropagation
        dA = -(np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon)) #for the L layer, derivative of the Loss function
        #NOTE: there might be an error when calculating the dZL (output layer) for the sigmoid activation. 
        # The course always uses dZ3 = Al - Y. 
        # Using the above formula for dAL and the the derivative of DZ(sigmoid) should return the same result, but there seems to be a small error...
        
        for l in reversed(range (L)):
            grads["dA" + str(l+1)] = dA #cache the dA passed from the previous layer

            if (l == (L-1)):
                A = AL
            else:
                A = caches[l+1][0][0]   #A from the linear cache

            dZ, dA, dW, db = DeepNN.linear_activation_backward(dA, A, caches[l], layers[l]["activation"], l2_reg_lambda)
            
            grads["dZ" + str(l+1)] = dZ
            grads["dW" + str(l+1)] = dW
            grads["db" + str(l+1)] = db


        return grads

    @staticmethod
    def update_parameters(parameters, grads, learning_rate):
        L = len(parameters) // 2 #because parameters are pairs of W[i], b[i] per layer

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters

    @staticmethod
    def update_parameters_adam(parameters, grads, learning_rate, adam_hyperparams, adam_parameters):
        L = len(parameters) // 2

        beta1 = adam_hyperparams.beta1
        beta2 = adam_hyperparams.beta2
        epsilon = adam_hyperparams.epsilon
        t = adam_parameters["t"] #adam counter
        v = adam_parameters["v"]
        s = adam_parameters["s"]
        v_corrected = {}
        s_corrected = {}
        
        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads["db" + str(l+1)]
            # Compute bias-corrected first moment estimate
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-beta1**t)

            # Moving average of the squared gradients
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * np.square(grads["dW" + str(l+1)])
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * np.square(grads["db" + str(l+1)])
            # Compute bias-corrected second raw moment estimate
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-beta2**t)

            # Update parameters
            parameters["W" + str(l+1)] -= learning_rate * ( v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon))
            parameters["b" + str(l+1)] -= learning_rate * ( v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon))

        return parameters, adam_parameters

    @staticmethod
    def gradient_check(parameters, grads, layers, X, Y, epsilon = 1e-7, tolerance = 2e-7):
        L = len(layers)
        
        #set-up variables
        parameter_names = ["W", "b"]
        grad_approx = []
        grad_backprop = []

        for l in range(L):
            for parameter_name in parameter_names:
                parameter_name = parameter_name + str(l+1)
                grad_name = "d" + parameter_name
                parameter = parameters[parameter_name]
                grad = grads[grad_name]

                assert(parameter.shape == grad.shape)

                for i in range(parameter.shape[0]):
                    for j in range(parameter.shape[1]):
                        # Grad backprop
                        grad_backprop.append(grad[i][j])

                        # Grad approx
                        # Compute J_plus
                        parameters_plus = deepcopy(parameters)
                        parameters_plus[parameter_name][i][j] += epsilon
                        AL_plus, _ = DeepNN.forward_prop(X, parameters_plus, layers)
                        J_plus = DeepNN.compute_cost(AL_plus, Y)

                        # Compute J_minus
                        parameters_minus = deepcopy(parameters)
                        parameters_minus[parameter_name][i][j] -= epsilon
                        AL_minus, _ = DeepNN.forward_prop(X, parameters_minus, layers)
                        J_minus = DeepNN.compute_cost(AL_minus, Y)
                        
                        # Compute gradapprox
                        numerical_grad = (J_plus-J_minus) / (2*epsilon)
                        grad_approx.append(numerical_grad)

        #compare grads (gradapprox to backward propagation gradients) by computing difference
        grad_approx = np.array(grad_approx)
        grad_backprop = np.array(grad_backprop)
        difference = (np.linalg.norm(grad_backprop-grad_approx)) / (np.linalg.norm(grad_backprop) + np.linalg.norm(grad_approx))

        if (difference > tolerance):
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
        
        return difference

    def fit(self, X, Y, num_epocs=10000, mini_batch_size=64, learning_rate=0.1, l2_reg_lambda=0.0, gradient_check=False, adam_hyperparams=None):
        costs = []                         # keep track of cost

        use_adam = adam_hyperparams is not None
        seed = 10        #NOTE: hardcoded seed

        parameters = DeepNN.initialize_parameters(X, self.layers)
        if (use_adam):
            adam_parameters = DeepNN.initialize_adam(parameters)

         # Loop (gradient descent)
        for i in range(num_epocs):

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1 #have a different seed for each shuffle. Hardcoded for debugging purposes
            X_shuffle, Y_shuffle = shuffle_data_set(X, Y, seed=seed)
            minibatches = split_data_set_to_mini_batches(X_shuffle, Y_shuffle, mini_batch_size)

            for minibatch in minibatches:
                 # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                #Forward prop
                AL, caches = DeepNN.forward_prop(minibatch_X, parameters, self.layers)
                
                #Cost
                cost = DeepNN.compute_cost(AL, minibatch_Y, parameters, l2_reg_lambda)
                
                #Back Prop
                grads = DeepNN.backward_prop(minibatch_X, minibatch_Y, AL, caches, self.layers, l2_reg_lambda)

                #Grad Check
                if (gradient_check):
                    DeepNN.gradient_check(parameters, grads, self.layers, minibatch_X, minibatch_Y)

                #Update Parameters
                if (use_adam):
                    adam_parameters["t"] += 1 #Increase Adam update counter
                    parameters, adam_parameters = DeepNN.update_parameters_adam(parameters, grads, learning_rate, adam_hyperparams, adam_parameters)
                else:
                    parameters = DeepNN.update_parameters(parameters, grads, learning_rate)
            
            costs.append(cost)
            if i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        self.parameters = parameters
        return costs

    def predict(self, X):
        AL, caches = DeepNN.forward_prop(X, self.parameters, self.layers)
        Y_pred = (AL > 0.5) * 1
        return Y_pred

    def load_parameters(self, parameters):
        self.parameters = parameters

#--------- Adam Hyper Parameters ---------#
class AdamHyperParams:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon


from testCases_reg import *
def grad_check_test():
    from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
    from gc_utils import forward_propagation_n, backward_propagation_n, gradient_check_n

    X, Y, parameters = gradient_check_n_test_case()
    layers = [{"n":5,      "activation":"relu",        "dropout_keep_prob":1},
              {"n":3,      "activation":"relu",        "dropout_keep_prob":1}, 
              {"n":1,      "activation":"sigmoid",     "dropout_keep_prob":1}]

    cost, cache = forward_propagation_n(X, Y, parameters)
    nn_AL, nn_caches = DeepNN.forward_prop(X, parameters, layers)

    gradients = backward_propagation_n(X, Y, cache)
    nn_grads = DeepNN.backward_prop(X, Y, nn_AL, nn_caches, layers)
    
    difference = gradient_check_n(parameters, gradients, X, Y)
    difference = gradient_check_n(parameters, nn_grads, X, Y)
    DeepNN.gradient_check(parameters, nn_grads, layers, X, Y)
    
#--------- TF DeepNN ---------#
class TFDeepNN():
    def __init__(self, layers):
        self.layers = layers
        self.graph = tf.Graph()

    def __del__(self):
        self.session.close()
    
    @staticmethod
    def initialize_parameters(X, layers, random_seed=None):
        if (random_seed is not None):
            tf.set_random_seed(random_seed)

        all_layers = layers.copy()
        all_layers.insert(0, {"n":X.shape[0]}) #add the inputs as the first layer

        parameters = {}
        L = len(all_layers)

        for l in range(1, L):
            parameters['W' + str(l)] = tf.get_variable('W' + str(l), [all_layers[l]["n"], all_layers[l-1]["n"]], initializer = tf.contrib.layers.xavier_initializer(seed=random_seed))
            parameters['b' + str(l)] = tf.get_variable('b'  + str(l), [all_layers[l]["n"], 1], initializer = tf.zeros_initializer())

        return parameters

    @staticmethod
    def forward_prop(X, parameters, layers):
        caches = []
        A = X
        L = len(layers)

        for l in range (L-1):
            Z = tf.add(tf.matmul(parameters['W' + str(l+1)], A), parameters['b' + str(l+1)])
            A = tf.nn.relu(Z)
        
        ZL = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
        return ZL

    @staticmethod
    def compute_cost(ZL, Y):
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(ZL)
        labels = tf.transpose(Y)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return cost

    def fit(self, X_train, Y_train, num_epochs=10000, mini_batch_size=64, learning_rate=0.1, random_seed=None):
        if (random_seed is not None):
            tf.set_random_seed(random_seed)

        min_batch_seed = 3        #NOTE: hardcoded seed

        #ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.reset_default_graph()
        with self.graph.as_default():

            (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
            n_y = Y_train.shape[0]                            # n_y : output size
            self.C = n_y
            costs = []                                        # To keep track of the cost

            # Create Placeholders of shape (n_x, n_y)
            X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
            Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")

            self.parameters = TFDeepNN.initialize_parameters(X, layers=self.layers, random_seed=random_seed)

            # Forward propagation: Build the forward propagation in the tensorflow graph
            ZL = TFDeepNN.forward_prop(X, self.parameters, self.layers)
            cost = TFDeepNN.compute_cost(ZL, Y)

            # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

             # Initialize all the variables
            init = tf.global_variables_initializer()

        # Run the initialization
        self.session = tf.Session(graph=self.graph)
        self.session.run(init) # Start the session to compute the tensorflow graph

        # Do the training
        for i in range(num_epochs):
            epoch_cost = 0

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            min_batch_seed = min_batch_seed + 1 #have a different seed for each shuffle. Hardcoded for debugging purposes
            X_shuffle, Y_shuffle = shuffle_data_set(X_train, Y_train, seed=min_batch_seed)
            minibatches = split_data_set_to_mini_batches(X_shuffle, Y_shuffle, mini_batch_size)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = self.session.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})

                epoch_cost += minibatch_cost
            
            epoch_cost = epoch_cost / (len(minibatches)) #average
            costs.append(epoch_cost)
            if (i % 100 == 0)  or (i == num_epochs - 1):
                print ("Cost after epoch %i: %f" %(i, epoch_cost))
        
        return costs

    def predict(self, X_data):
        (n_x, m) = X_data.shape

        with self.graph.as_default():
            X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
            ZL = TFDeepNN.forward_prop(X, self.parameters, self.layers)
            predict = tf.argmax(ZL)

        pred = predict.eval({X: X_data}, session=self.session)
        return pred
    
    def calc_accuracy(self, X_data, Y_data):
        (n_x, m) = X_data.shape
        n_y = Y_data.shape[0]

        with self.graph.as_default():
            X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
            Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")

            ZL = TFDeepNN.forward_prop(X, self.parameters, self.layers)
            correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accuracy_value = accuracy.eval({X: X_data, Y: Y_data}, session=self.session)

        return accuracy_value

    def eval_value(self, value):
        return value.eval(session=self.session)

if __name__ == "__main__":

    #change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    """
    #---------- WEEEK 1 ----------#
    #Loading the data
    datasets = py_datasets.load_week3_datasets(500)

    #Get planar dataset
    X, Y = datasets["planar"]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    #plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
    #plt.show()

    np.random.seed(1)
    X = feature_standardization(X)
    X, Y = suffle_data_set(X, Y)

    split_buckets = [0.8, 0.2]
    [train_set_x, test_set_x] = split_data_set(X, split_buckets)
    [train_set_y, test_set_y] = split_data_set(Y, split_buckets)

    #DeepNN
    layers = [{"n":20,      "activation":"relu",        "dropout_keep_prob":1.0},
              #{"n":10,      "activation":"relu",        "dropout_keep_prob":1},    
              {"n":1,      "activation":"sigmoid",     "dropout_keep_prob":1}]
    deep_nn = DeepNN(layers = layers)
    #train
    costs = deep_nn.fit(train_set_x, train_set_y, num_iterations=10000, learning_rate=0.4, l2_reg_lambda=0.4, gradient_check=False)

    plt.plot(costs)
    plt.show()

    #predict
    y_pred_train = deep_nn.predict(train_set_x)
    y_pred_test = deep_nn.predict(test_set_x)

    # Print train/test Errors= 
    train_accuracy = calc_accuracy(y_pred_train, train_set_y)
    test_accuracy = calc_accuracy(y_pred_test, test_set_y)
    print("train accuracy: {} %".format(train_accuracy))
    print("test accuracy: {} %".format(test_accuracy))

    #plot decision boundaries
    plot_decision_boundary(lambda x: deep_nn.predict(x.T), train_set_x, train_set_y)
    plt.title("Train Data, A:{}%".format(train_accuracy))
    plt.show()

    plot_decision_boundary(lambda x: deep_nn.predict(x.T), train_set_x, train_set_y)
    plt.title("Test Data, A:{}%".format(test_accuracy))
    plt.show()

    #---Grad Check Test---
    grad_check_test()
    """

    """
    #---------- WEEEK 2----------#
    from testCases_opt import *
    import opt_utils

    #--- Minibatch ---
    X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    X_shuffle, Y_shuffle = shuffle_data_set(X_assess, Y_assess, seed=0)
    mini_batches = split_data_set_to_mini_batches(X_shuffle, Y_shuffle, mini_batch_size)

    print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
    print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
    
    #--- Update params with Adam Opt ---
    parameters, grads, v, s = update_parameters_with_adam_test_case()
    adam_params = {}
    adam_params["v"] = v
    adam_params["s"] = s
    adam_params["t"] = 2

    adam_hyper = AdamHyperParams()
    learning_rate = 0.01
    
    parameters, adam_params = DeepNN.update_parameters_adam(parameters, grads, learning_rate, adam_hyper, adam_params)

    v = adam_params["v"]
    s = adam_params["s"]

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))

    #--- DeepNN model with Adam ---

    train_X, train_Y = opt_utils.load_dataset()


    #DeepNN
    layers = [{"n":5,      "activation":"relu",        "dropout_keep_prob":1.0},
              {"n":2,      "activation":"relu",        "dropout_keep_prob":1.0},    
              {"n":1,      "activation":"sigmoid",     "dropout_keep_prob":1.0}]

    deep_nn_adam = DeepNN(layers = layers)
    #train
    costs = deep_nn_adam.fit(train_X, train_Y, learning_rate=0.0007, adam_hyperparams=AdamHyperParams())
    plt.plot(costs)
    plt.show()

    #plot decision boundaries
    plot_decision_boundary(lambda x: deep_nn_adam.predict(x.T), train_X, train_Y)
    plt.show()
    """

    #"""
    #---------- WEEEK 3----------#

    import tf_utils
    # Loading the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()
    C = 6

    # Example of a picture
    index = 200
    #plt.imshow(X_train_orig[index])
    #plt.show()
    print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

    X_train_flatten = flatten_matrix(X_train_orig)
    X_test_flatten = flatten_matrix(X_test_orig)

    X_train = X_train_flatten/255.0 #
    X_test = X_test_flatten/255.0
    #X_train = feature_standardization(X_train_flatten)
    #X_test = feature_standardization(X_test_flatten)

    Y_train = convert_to_one_hot(Y_train_orig, C)
    Y_test = convert_to_one_hot(Y_test_orig, C)

    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    ##--TensorFlow DNN--##
    layers = [{"n":25},
              {"n":12},    
              {"n":C}]

    tf_dnn = TFDeepNN(layers = layers)
    costs = tf_dnn.fit(X_train, Y_train, num_epochs=100, mini_batch_size=32, learning_rate=0.001, random_seed=1)
    plt.plot(costs)
    plt.show()

    #predict
    print ("Train Accuracy:", tf_dnn.calc_accuracy(X_train, Y_train))
    print ("Test Accuracy:", tf_dnn.calc_accuracy(X_test, Y_test))
















