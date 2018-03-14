#https://www.tensorflow.org/get_started/input_fn

#no clue what these do and are for....?!?!??
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import itertools

import pandas as pd
import tensorflow as tf

#util function
def get_input_fn(data_set, features, label, num_epochs=None, shuffle=True):
#using pandas dataframe
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in features}),
        y = pd.Series(data_set[label].values),
        num_epochs=num_epochs,
        shuffle=shuffle)

def boston_housing():
    #load data
    COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
    FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
    LABEL = "medv"

    training_set = pd.read_csv("boston_housing_dataset/boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("boston_housing_dataset/boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("boston_housing_dataset/boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

    #defining feature columns
        #MORE INFO: https://www.tensorflow.org/tutorials/linear#feature_columns_and_transformations
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES] #this is enough because all features are numerical countinous values
    
    #create the regressor
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir="/tmp/boston_model")

    #Building the input_fn!!! -> get_input_fn

    #training
    regressor.train(input_fn=get_input_fn(training_set, FEATURES, LABEL), steps=5000)

    #evalutate
    ev = regressor.evaluate(input_fn=get_input_fn(test_set, FEATURES, LABEL, num_epochs=1, shuffle=False))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    #make predictions
    y_ = regressor.predict(input_fn=get_input_fn(prediction_set, FEATURES, LABEL, num_epochs=1, shuffle=False))
    predictions = list(p["predictions"] for p in itertools.islice(y_, 6)) #6 predictions
    print("Predictions: {}".format(str(predictions)))

    


if __name__ == "__main__":
    #change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #set logging verbosity to INFO (more detailed)
    tf.logging.set_verbosity(tf.logging.INFO) 

    boston_housing()

    
   