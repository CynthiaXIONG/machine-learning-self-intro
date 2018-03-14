#https://www.tensorflow.org/get_started/mnist/pros

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def simple_one_layer_nn():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("\n Simple-One-layer Test Accuracy: {0:f}\n".format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

def multilayer_convolutional_nn():
    print("TODO!!")

def tf_estimator_nn():
    #based on the oficial tutorial and https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940
    mnist = input_data.read_data_sets('MNIST_data') #DNNClassifier estimator does not support one hot encoded labels...
    # Specify features
    feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

    # Build 2 layer DNN with 25, 12 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[256, 32],
                                            optimizer=tf.train.AdamOptimizer(1e-4),
                                            n_classes=10,
                                            dropout=0.1,
                                            model_dir="/tmp/mnist_model") #The directory in which TensorFlow will save checkpoint data and TensorBoard summaries during model training
    
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(mnist.train.images)},
        y=np.array((mnist.train.labels).astype(np.int32)),
        num_epochs=None,
        shuffle=True)
    
    #if pandas
    #my_input_fn = tf.estimator.inputs.pandas_input_fn(
    #x=pd.DataFrame({"x": x_data}),
    #y=pd.Series(y_data),
    #...)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=10000)

    #Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(mnist.test.images)},
        y=np.array((mnist.test.labels).astype(np.int32)),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\n Estimator Test Accuracy: {0:f}\n".format(accuracy_score))

if __name__ == "__main__":
    #change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    simple_one_layer_nn()

    multilayer_convolutional_nn()

    tf_estimator_nn()
   