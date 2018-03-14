#code style: https://google.github.io/styleguide/pyguide.html
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

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

#--------- DATA PROCESSING ---------#
def flatten_matrix(m):
    dim_size = 1
    for i in range(1, len(m.shape)):
        dim_size *= m.shape[i] 

    return m.reshape(-1, dim_size).T

def normalization(x, feature_axis = 1):
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord = None, axis = feature_axis, keepdims = True)
    # Divide x by its norm.
    return x / x_norm

def min_max_scalling(x, feature_axis = 1):
    # Compute x_min and x_max
    x_min = np.min(x, axis = feature_axis, keepdims = True)
    x_max = np.max(x, axis = feature_axis, keepdims = True)
    # Scale
    return (x -x_min) / (x_max - x_min)

def feature_standardization(x, feature_axis = 1):
    # https://en.wikipedia.org/wiki/Feature_scaling
    # Subtract the mean (to have zero mean) and devide by the std
    return (x - np.mean(x, axis = feature_axis, keepdims = True)) / np.std(x, axis = feature_axis, keepdims = True)

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
def sigmoid_derivative(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def tanh_derivative(dA, Z):
    return 1 #1 - np.power(np.tanh(z), 2)  <- wrong?...needs dA?

def relu_derivative(dA, Z):
    return np.multiply((Z > 0), dA)

def leaky_relu_derivative(dA, Z):
    positive = np.multiply((Z > 0), dA)
    negative = np.multiply((Z <= 0), 0.01 * dA)
    return positive + negative

#--------- PARAMETER INITIALIZATION ---------#
def initialize_parameters_with_zero(dim):
    w = np.zeros([dim, 1])
    b = 0
    return w, b

#--------- PERFORMANCE METRIC ---------#
def calc_accuracy(y_predict, y):
    return 1 - np.mean(np.abs(y_predict - y))

#--------- NEURAL NET BASIC (Logistic Regression) ---------#
class BasicNN():
    def __init__(self):
        pass

    @staticmethod
    def calc_A(w, b, X):
        return sigmoid(np.dot(w.T, X) + b)
    
    @staticmethod
    def propagate(w, b, X, Y):
        # Implement the cost function and its gradient for the propagation
        
        m = Y.shape[1]

        # Forward Prop
        A = BasicNN.calc_A(w, b, X)
        cost = -(1/m) * (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T))

        # Back Prop
        dz = A-Y
        dw = (1/m) * np.dot(X, dz.T)
        db = (1/m) * np.sum(dz)

        grads = {"dw": dw,
                 "db" : db}

        cost = np.squeeze(cost) #convert to scalar from matrix (1,1)

        return grads, cost

    def fit(self, X, Y, num_iterations, learning_rate):
        #    This function osptimizes w and b by running a gradient descent algorithm

        # Initialize parameters
        n = X.shape[0]
        w, b = initialize_parameters_with_zero(n)

        costs = []
        for i in range(num_iterations):
            #compute graph to get the grads and cost
            grads, cost = BasicNN.propagate(w, b, X, Y)
            
            #update rule
            w -= learning_rate * grads["dw"]
            b -= learning_rate * grads["db"]

            #append cost
            costs.append(cost)
        
        self.params = {"w": w,
                       "b": b}

        return costs

    def predict(self, X):
        A = BasicNN.calc_A(self.params["w"], self.params["b"], X)
        Y_pred = (A > 0.5) * 1
        return Y_pred

#--------- NEURAL NET SIMPLE (2 layers) ---------#
class SimpleNN():
    def __init__(self, hidden_layer_size, w_initialization_factor = 0.01):
        self.n_h = hidden_layer_size
        self.w_init_factor = w_initialization_factor

    def random_initialize_parameters(self):
        W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        b1 = np.zeros([self.n_h, 1])
        W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        b2 = np.zeros([self.n_y, 1])

        self.parameters = {"W1": W1,
                           "b1": b1,
                           "W2": W2,
                           "b2": b2}

    @staticmethod
    def forward_propagation(X, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2 }

        return A2, cache

    def propagate(self, X, Y):
        # Implement the cost function and its gradient for the propagation
        m = Y.shape[1]
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        # Forward Prop
        A2, cache = SimpleNN.forward_propagation(X, self.parameters)
        A1 = cache["A1"]

        cost = -(1/m) * np.sum((np.dot(Y, np.log(A2).T) + np.dot((1-Y), np.log(1-A2).T)))
        cost = np.squeeze(cost)

        # Back Prop
        dZ2 = A2-Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1-np.power(A1, 2))) 
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads, cost

    def fit(self, X, Y, num_iterations, learning_rate):
        #    This function osptimizes w and b by running a gradient descent algorithm

        # Initialize parameters
        self.n_x = X.shape[0]
        self.n_y = Y.shape[0]
        self.random_initialize_parameters()

        costs = []
        for i in range(num_iterations):
            #compute graph to get the grads and cost
            grads, cost = self.propagate(X, Y)
            
            #update rule
            self.parameters["W1"] -= learning_rate * grads["dW1"]
            self.parameters["b1"] -= learning_rate * grads["db1"]
            self.parameters["W2"] -= learning_rate * grads["dW2"]
            self.parameters["b2"] -= learning_rate * grads["db2"]

            #append cost
            costs.append(cost)

            if i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return costs

    def predict(self, X):
        A2, cache = SimpleNN.forward_propagation(X, self.parameters)
        Y_pred = (A2 > 0.5) * 1
        return Y_pred


#--------- NEURAL NET SIMPLE (2 layers) ---------#
class DeepNN():
    def __init__(self, layers, w_initialization_factor = 0.01):
        self.layers = layers
        self.w_init_factor = w_initialization_factor

    @staticmethod
    def initialize_parameters(layers):
        parameters = {}
        L = len(layers)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers[l]["n"], layers[l-1]["n"]) / np.sqrt(layers[l-1]["n"]) #*w_initialization_factor
            parameters['b' + str(l)] = np.zeros((layers[l]["n"], 1))

        return parameters

    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        linear_cache = (A_prev, W, b)
        Z = W.dot(A_prev) + b
        
        activation_cache = Z
        if activation == "sigmoid":
            A = sigmoid(Z) 
        elif activation == "tanh":
            A = tanh(Z)
        elif activation == "relu":
            A = relu(Z)
        elif activation == "leaky_relu":
            A = leaky_relu(Z)
        
        cache = (linear_cache, activation_cache)
        return A, cache

    @staticmethod
    def forward_prop(X, parameters, layers):
        caches = []
        A = X
        L = len(layers)

        for l in range (1, L):
            A_prev = A
            A, cache = DeepNN.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation=layers[l]["activation"])
            caches.append(cache)
        
        return A, caches

    @staticmethod
    def compute_cost(AL, Y):
        m = Y.shape[1]
        cost = -(1/m) * np.sum((np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T)))

        assert(cost.shape == ())
        return np.squeeze(cost)

    @staticmethod
    def linear_activation_backward(dA, cache, activation):
        #get cached values
        linear_cache, activation_cache = cache
        A_prev, W, b = linear_cache
        Z = activation_cache

        #calc dZ from derivative of the activation function, dA
        if activation == "sigmoid":
            dZ = sigmoid_derivative(dA, Z) 
        elif activation == "tanh":
            dZ = tanh_derivative(dA, Z) 
        elif activation == "relu":
            dZ = relu_derivative(dA, Z) 
        elif activation == "leaky_relu":
            dZ = leaky_relu_derivative(dA, Z) 

        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)

        return dA_prev, dW, db

    @staticmethod
    def backward_prop(AL, Y, caches, layers):
        L = len(layers)
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        grads = {}

        # Initializing the backpropagation
        dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) #for the L layer, derivative of the Loss function
        
        for l in reversed(range (L-1)):
            dA, dW, db = DeepNN.linear_activation_backward(dA, caches[l], activation=layers[l+1]["activation"])
            grads["dA" + str(l+1)] = dA
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

    def fit(self, X, Y, num_iterations, learning_rate):
        costs = []                         # keep track of cost

        layers.insert(0, {"n":X.shape[0]})
        parameters = DeepNN.initialize_parameters(self.layers)

         # Loop (gradient descent)
        for i in range(0, num_iterations):
            #Forward prop
            AL, caches = DeepNN.forward_prop(X, parameters, self.layers)
            
            #Cost
            cost = DeepNN.compute_cost(AL, Y)
            costs.append(cost)
            if i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            
            #Back Prop
            grads = DeepNN.backward_prop(AL, Y, caches, self.layers)

            #Update Parameters
            parameters = DeepNN.update_parameters(parameters, grads, learning_rate)

        self.parameters = parameters
        return costs

    def predict(self, X):
        AL, caches = DeepNN.forward_prop(X, self.parameters, self.layers)
        Y_pred = (AL > 0.5) * 1
        return Y_pred

if __name__ == "__main__":

     #change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #---------- WEEEK 2 ----------#
    """
    #Loading the data (cat/non-cat)
    train_data_path = os.path.abspath('datasets/train_catvnoncat.h5')
    test_data_path = os.path.abspath('datasets/test_catvnoncat.h5')
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_train_test_dataset(train_data_path, test_data_path)

    #show img
    plt.imshow(train_set_x_orig[25])
    plt.show()

    #Basic NN (Logistic Regression)
    basic_nn = BasicNN()

    #process input
    flatten_train_set_x = flatten_matrix(train_set_x_orig)
    flatten_test_set_x = flatten_matrix(test_set_x_orig)

    #train_set_x = flatten_train_set_x / 255
    #test_set_x = flatten_test_set_x / 255
    #train_set_x = normalize_rows(flatten_train_set_x)
    #test_set_x = normalize_rows(flatten_test_set_x)
    #train_set_x = min_max_scalling(flatten_train_set_x)
    #test_set_x = min_max_scalling(flatten_test_set_x)
    train_set_x = feature_standardization(flatten_train_set_x)
    test_set_x = feature_standardization(flatten_test_set_x)

    #train
    costs = basic_nn.fit(train_set_x, train_set_y, num_iterations=2000, learning_rate=0.005)

    plt.plot(costs)
    plt.show()

    #predict
    y_pred_train = basic_nn.predict(train_set_x)
    y_pred_test = basic_nn.predict(test_set_x)

    # Print train/test Errors
    print("train accuracy: {} %".format(calc_accuracy(y_pred_train, train_set_y)))
    print("test accuracy: {} %".format(calc_accuracy(y_pred_test, test_set_y)))

    #Test one example
    for i in range(1, 5):
        plt.imshow(flatten_test_set_x[:, i].reshape((test_set_x_orig.shape[1], test_set_x_orig.shape[2], test_set_x_orig.shape[3])))
        plt.title("y = " + str(test_set_y[0, i]) + ", you predicted that it is a \"" + classes[y_pred_test[0, i]].decode("utf-8") +  "\" picture.")
        plt.show()
    """
    #---------- WEEEK 3 ----------#
    
    datasets = py_datasets.load_week3_datasets(500)

    #Get planar dataset
    X, Y = datasets["planar"]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
    #plt.show()

    # Train the logistic regression (BasicNN)
    basic_nn = BasicNN()
    costs = basic_nn.fit(X, Y, num_iterations=10000, learning_rate=1.2)
    plot_decision_boundary(lambda x: basic_nn.predict(x.T), X, Y)
    plt.title("Logistic Regression")
    plt.show()

    simple_nn = SimpleNN(5)
    costs = simple_nn.fit(X, Y, num_iterations=10000, learning_rate=1.2)
    plot_decision_boundary(lambda x: simple_nn.predict(x.T), X, Y)
    plt.title("Shallow NN")
    plt.show()
    
    layers = [{"n":5,      "activation":"sigmoid"},  
              {"n":1,       "activation":"sigmoid"}]
    deep_nn = DeepNN(layers = layers)
    costs = deep_nn.fit(X, Y, num_iterations=10000, learning_rate=1.2)
    plot_decision_boundary(lambda x: deep_nn.predict(x.T), X, Y)
    plt.title("Deep NN")
    plt.show()

    """
    #---------- WEEEK 4 ----------#
    #Loading the data (cat/non-cat)
    train_data_path = os.path.abspath('datasets/train_catvnoncat.h5')
    test_data_path = os.path.abspath('datasets/test_catvnoncat.h5')
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_train_test_dataset(train_data_path, test_data_path)

    #DeepNN
    np.random.seed(1)
    layers = [{"n":20,      "activation":"relu"}, 
              {"n":7,       "activation":"relu"}, 
              {"n":5,       "activation":"relu"}, 
              {"n":1,       "activation":"sigmoid"}]
    deep_nn = DeepNN(layers = layers)

    #process input
    flatten_train_set_x = flatten_matrix(train_set_x_orig)
    flatten_test_set_x = flatten_matrix(test_set_x_orig)

    train_set_x = flatten_train_set_x / 255
    test_set_x = flatten_test_set_x / 255
    #train_set_x = normalization(flatten_train_set_x)
    #test_set_x = normalization(flatten_test_set_x)
    #train_set_x = min_max_scalling(flatten_train_set_x)
    #test_set_x = min_max_scalling(flatten_test_set_x)
    #train_set_x = feature_standardization(flatten_train_set_x)
    #test_set_x = feature_standardization(flatten_test_set_x)

    #train
    costs = deep_nn.fit(train_set_x, train_set_y, num_iterations=3001, learning_rate=0.0075)

    plt.plot(costs)
    plt.show()

    #predict
    y_pred_train = deep_nn.predict(train_set_x)
    y_pred_test = deep_nn.predict(test_set_x)

    # Print train/test Errors
    print("train accuracy: {} %".format(calc_accuracy(y_pred_train, train_set_y)))
    print("test accuracy: {} %".format(calc_accuracy(y_pred_test, test_set_y)))
    """
    