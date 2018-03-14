import os
import sys
import io

# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Deep Learning 
import tensorflow as tf

custom_path = os.path.abspath('D:/Git Projects/bot-winter-project/study material/cousera deeplearning/Course 2 - Improving NN/scripts')
if custom_path not in sys.path:
    sys.path.append(custom_path)
import coursera_dp_master

custom_path = os.path.abspath('D:/Git Projects/bot-winter-project/kaggle/introduction/Titanic Machine Learning from Disaster')
if custom_path not in sys.path:
    sys.path.append(custom_path)
import titanic_data_science_solutions

#Utils
def visualize_image(pixels_array, image_size, cmap_name="gray"):
    assert(pixels_array.size == (image_size[0] * image_size[1]))
    pixels_matrix = pixels_array.reshape(image_size[0], image_size[1])
    plt.imshow(pixels_matrix, cmap=plt.get_cmap(cmap_name))

def convert_to_categorical(serie):
    return pd.get_dummies(serie)

def first_attempt():
     ## Aquire Data ##
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')
    image_size = [28, 28]

    print(train_df.shape)

    X_train_o = train_df.drop(["label"], axis=1).values.astype('float32') # all pixel values
    y_train_o = train_df["label"].values.astype('int32') # only labels i.e targets digits
    X_test_o = test_df.values.astype('float32')

    #visualize some pics
    for i in range(6, 9):
        plt.subplot(330 + (i+1))
        visualize_image(X_train_o[i], image_size)
        plt.title(y_train_o[i])
    plt.show()

    #preprocess data
    X_train = X_train_o.T
    X_test = X_test_o.T

        #feature scalling
    coursera_dp_master.feature_standardization(X_train, axis=1)
    coursera_dp_master.feature_standardization(X_test, axis=1)

        #one-hot encondig
    y_train = convert_to_categorical(y_train_o).values
    y_train = y_train.T
    C = y_train.shape[0]

    # fix random seed for reproducibility
    seed = 1
    np.random.seed(seed)

     ##--TensorFlow DNN--##
    layers = [{"n":25},
              {"n":12},    
              {"n":C}]
    print (X_train.shape, y_train.shape)

    tf_dnn = coursera_dp_master.TFDeepNN(layers = layers)
    costs = tf_dnn.fit(X_train, y_train, num_epochs=1000, mini_batch_size=64, learning_rate=0.001, random_seed=seed)
    plt.plot(costs)
    plt.show()

    #predict
    y_train_pred = tf_dnn.predict(X_train)
    y_test = tf_dnn.predict(X_test)

     # Print train Errors= 
    train_accuracy = coursera_dp_master.calc_accuracy(y_train_pred, y_train_o)
    print("train accuracy: {} %".format(train_accuracy))

    submission = pd.DataFrame({
        "ImageId": list(range(1, len(y_test)+1)),
        "Label": y_test.flatten()
    })
    submission.to_csv('output/submission0.csv', index=False)


if __name__ == "__main__":
    #change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    first_attempt()

    
