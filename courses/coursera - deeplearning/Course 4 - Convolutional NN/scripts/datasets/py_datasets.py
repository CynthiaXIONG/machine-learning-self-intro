import os
import numpy as np
import h5py

def load_signs_dataset():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

### UTILS ###

def shuffle_data_set(X, Y, seed=None):
    if (seed is not None):
        np.random.seed(seed)            # To make the "random shuffle" the same, for testing purposes

    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    return shuffled_X, shuffled_Y

def split_data_set_to_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[0]
    mini_batches = []

    # Partition the mini-batches
    num_minibatches = math.ceil(m/mini_batch_size)
    for k in range(0, num_minibatches):
        mini_batch_X = X[mini_batch_size*k : min(mini_batch_size*(k+1), m), :, :, :]
        mini_batch_Y = Y[mini_batch_size*k : min(mini_batch_size*(k+1), m), :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_barches(X, Y, mini_batch_size=64, seed=0):
    X_shuffle, Y_shuffle = shuffle_data_set(X, Y, seed=seed)
    mini_batches = split_data_set_to_mini_batches(X_shuffle, Y_shuffle, mini_batch_size)
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

