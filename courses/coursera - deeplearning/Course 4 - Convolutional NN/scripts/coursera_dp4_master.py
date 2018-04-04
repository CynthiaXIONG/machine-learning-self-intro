#code style: https://google.github.io/styleguide/pyguide.html
import os
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import h5py

import datasets.py_datasets as py_datasets

import tensorflow as tf
from tensorflow.python.framework import ops

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

import pydot
#from IPython.display import SVG

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

#--------- CONV UTILS ---------#
def pad_image(X, padding, value=0):
    num_dim = len(X.shape)
    assert(num_dim == 3 or num_dim == 4)

    if (num_dim == 3):
        return np.pad(X, ((padding, padding), (padding, padding), (0, 0)), "constant", constant_values = (value, value))

    else:
        return np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), "constant", constant_values = (value, value))
    
def unpad_image(X, padding):
    num_dim = len(X.shape)
    assert(num_dim == 3 or num_dim == 4)

    if (num_dim == 3):
        return X[padding:-padding, padding:-padding, :]

    else:
        return X[:, padding:-padding, padding:-padding, :]

def calc_conv_output_dim(n_prev, f=1, pad=0, stride=1):
    return int((n_prev - f + 2 * pad) / stride) + 1

def create_max_mask(x): 
    #for max pool backprop
    mask = (x == np.max(x))
    return mask

def distribute_value_mask(dz, shape):
    #for average pool backprop
    average = dz / np.prod(shape)
    a = np.ones(shape) * average
    return a

#--------- CONV DEEP NET (1 layer - Basic Implementation) ---------#
class ConvDeep1NN():
    def __init__(self):
        pass

    @staticmethod
    def _get_slide_window_corners(f, stride, h, w):
        vert_start = h*stride
        vert_end = vert_start + f
        horiz_start = w*stride
        horiz_end = horiz_start + f

        return vert_start, vert_end, horiz_start, horiz_end

    @staticmethod
    def _conv_single_step(a_slice, W, b):
        Z = np.sum(np.multiply(a_slice, W)) + float(b)
        return Z

    @staticmethod
    def _conv_forward(A_prev, W, b, hparameters):
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters"
        stride = hparameters["stride"]
        pad = hparameters["pad"]

        # Compute the dimensions of the CONV output volume
        n_H = calc_conv_output_dim(n_H_prev, f, pad, stride)
        n_W = calc_conv_output_dim(n_W_prev, f, pad, stride)

         # Initialize the output volume Z with zeros
        Z = np.zeros([m, n_H, n_W, n_C])

        # Create A_prev_pad by padding A_prev
        A_prev_pad = pad_image(A_prev, pad)

        for i in range(m):                               # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]                               # Select ith training example's padded activation
            for h in range(n_H):                           # loop over vertical axis of the output volume
                for w in range(n_W):                       # loop over horizontal axis of the output volume
                    for c in range(n_C):                   # loop over channels (= #filters) of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start, vert_end, horiz_start, horiz_end = ConvDeep1NN._get_slide_window_corners(f, stride, h, w)
                        
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = ConvDeep1NN._conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
            
        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))
        
        # Save information in "cache" for the backprop
        cache = (A_prev, W, b, hparameters)
        return Z, cache

    @staticmethod
    def _pool_forward(A_prev, hparameters, mode="max"):
         # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]
        
        # Define the dimensions of the output
        n_H = calc_conv_output_dim(n_H_prev, f=f, stride=stride)
        n_W = calc_conv_output_dim(n_W_prev, f=f, stride=stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):                         # loop over the training examples
            a_prev = A_prev[i] 
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start, vert_end, horiz_start, horiz_end = ConvDeep1NN._get_slide_window_corners(f, stride, h, w)
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))
        
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)
        return A, cache

    @staticmethod
    def _conv_backward(dZ, cache):
    
        # Retrieve information from "cache"
        (A_prev, W, b, hparameters) = cache
        
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters"
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape
        
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros(A_prev.shape)                           
        dW =  np.zeros(W.shape)
        db = np.zeros(b.shape)

        # Pad A_prev and dA_prev
        A_prev_pad = pad_image(A_prev, pad)
        dA_prev_pad = pad_image(dA_prev, pad)
        
        for i in range(m):                       # loop over the training examples
            
            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i, :, :, :]
            da_prev_pad = dA_prev_pad[i, :, :, :]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice"
                        vert_start, vert_end, horiz_start, horiz_end = ConvDeep1NN._get_slide_window_corners(f, stride, h, w)
                        
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
                        
            # Set the ith training example's dA_prev to the unpaded da_prev_pad
            dA_prev[i, :, :, :] = unpad_image( da_prev_pad, pad)
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        return dA_prev, dW, db

    @staticmethod
    def _pool_backward(dA, cache, mode = "max"):
        # Retrieve information from cache
        (A_prev, hparameters) = cache
        
        # Retrieve hyperparameters from "hparameters"
        stride = hparameters["stride"]
        f = hparameters["f"]
        
        # Retrieve dimensions from A_prev's shape and dA's shape
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):                       # loop over the training examples
            
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i,:,:,:]
            
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start, vert_end, horiz_start, horiz_end = ConvDeep1NN._get_slide_window_corners(f, stride, h, w)
                        
                        # Compute the backward propagation in both modes.
                        if mode == "max":
                            # Use the corners and "c" to define the current slice from a_prev
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            # Create the mask from a_prev_slice
                            mask = create_max_mask(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                            
                        elif mode == "average":
                            # Get the value a from dA
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value_mask(da, shape)
    
        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)
        return dA_prev

#--------- CONV DEEP NET - KERAS ---------#
    ## -- UTILS -- ##
def rn_identity_block(X, f, filters, stage, block, seed=0):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path, filters of shape (1,1) and a stride of (1,1). Its padding is "valid"
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path, filters of shape  (f,f)(f,f)  and a stride of (1,1). Its padding is "same"
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path,  filters of shape (1,1) and a stride of (1,1). Its padding is "valid" 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def rn_convolutional_block(X, f, filters, stage, block, s = 2, seed=0):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path, filters of shape (1,1) and a stride of (s,s). Its padding is "valid"
    X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path, filters of (f,f) and a stride of (1,1). Its padding is "same"
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path, filters of (1,1) and a stride of (1,1). Its padding is "valid"
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=seed))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####,  filters of shape (1,1) and a stride of (s,s). Its padding is "valid" 
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=seed))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X =  Add()([X, X_shortcut])
    X = Activation('relu')(X)
 
    return X

## -- ResNet -- ##
def ResNet50(input_shape = (64, 64, 3), classes = 6):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    filters_size = np.array([64, 64, 256])
    X = rn_convolutional_block(X, f = 3, filters = filters_size, stage = 2, block='a', s = 1)
    X = rn_identity_block(X, 3, filters_size, stage=2, block='b')
    X = rn_identity_block(X, 3, filters_size, stage=2, block='c')

    # Stage 3
    filters_size *= 2
    X = rn_convolutional_block(X, f = 3, filters = filters_size, stage = 3, block='a', s = 2)
    X = rn_identity_block(X, 3, filters_size, stage=3, block='b')
    X = rn_identity_block(X, 3, filters_size, stage=3, block='c')
    X = rn_identity_block(X, 3, filters_size, stage=3, block='d')

    # Stage 4
    filters_size *= 2
    X = rn_convolutional_block(X, f = 3, filters = filters_size, stage = 4, block='a', s = 2)
    X = rn_identity_block(X, 3, filters_size, stage=4, block='b')
    X = rn_identity_block(X, 3, filters_size, stage=4, block='c')
    X = rn_identity_block(X, 3, filters_size, stage=4, block='d')
    X = rn_identity_block(X, 3, filters_size, stage=4, block='f')
    X = rn_identity_block(X, 3, filters_size, stage=4, block='e')

    # Stage 5
    filters_size *= 2
    X = rn_convolutional_block(X, f = 3, filters = filters_size, stage = 5, block='a', s = 2)
    X = rn_identity_block(X, 3, filters_size, stage=5, block='b')
    X = rn_identity_block(X, 3, filters_size, stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D((2, 2), strides=(1, 1))(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model

## -- Keras-LeNet5 -- ##
def KerasLeNet5(input_shape = (64, 64, 3), classes = 6):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    X = X_input #no intial zero padding

    #CONV1
    X = Conv2D(filters=6, kernel_size=(5, 5), strides = (1, 1), padding = 'valid', name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)
    #CONV2
    X = Conv2D(filters=16, kernel_size=(5, 5), strides = (1, 1), padding = 'valid', name = 'conv2', kernel_initializer=glorot_uniform(seed=0))(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)
    #FC3
    X = Flatten()(X)
    X = Dense(120, activation='relu', name='fc3')(X)
    #FC4 
    X = Dense(84, activation='relu', name='fc4')(X)
    #OUT-Softmax
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='LeNet5')
    return model


#--------- WEEK 1-A ---------#
def week_1_a():
    np.random.seed(1)

    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride" : 1, "f": 2}
    A, cache = ConvDeep1NN._pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = ConvDeep1NN._pool_backward(dA, cache, mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])  
    print()
    dA_prev = ConvDeep1NN._pool_backward(dA, cache, mode = "average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1]) 

#--------- WEEK 2 ---------#
def week_2():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = py_datasets.load_signs_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Convert training and test labels to one hot matrices
    Y_train = py_datasets.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = py_datasets.convert_to_one_hot(Y_test_orig, 6).T

    model = ResNet50(input_shape = (64, 64, 3), classes = 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
       
def try_lenet_5_keras():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = py_datasets.load_signs_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Convert training and test labels to one hot matrices
    Y_train = py_datasets.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = py_datasets.convert_to_one_hot(Y_test_orig, 6).T

    model = KerasLeNet5(input_shape = (64, 64, 3), classes = 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs = 100, batch_size = 32)

    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    #plot_model(model, to_file='model.png')

#--------- MAIN ---------#
def main():
    # Change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #week_1_a()

    week_2()

    #try_lenet_5_keras()





if __name__ == "__main__":
    main()

    










