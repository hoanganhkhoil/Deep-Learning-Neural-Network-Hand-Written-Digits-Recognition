# Author: Khoi Hoang
# Neural Network - Hand Written Digits Recognition

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# Create Placeholders
X = tf.placeholder(tf.float32, [None, 28, 28,1])     # 28 x 28 pixel, 1 input channel (gray scale)
Y_ = tf.placeholder(tf.float32, [None, 10])          # 10 output channels

# Create Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
                
# Flatten input into 1 vector (28 x 28 = 784)
XX = tf.reshape(X, [-1, 784])

# Hypothesis function - Softmax
Y = tf.nn.softmax(tf.matmul(XX,W) + b)

# Loss function - Cross_Entropy
Cross_Entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# Accuracy
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
Accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Create Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(Cross_Entropy)


init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# Train model
for i in range(10000):
    # Get batch data
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data = {X: batch_x, Y_: batch_y}

    # train
    session.run(train_step, feed_dict=train_data)

    if i % 1000 == 0:
        C,A = session.run([Cross_Entropy, Accuracy], feed_dict=train_data)
        print ("Epoch: %s, Loss: %s, Accuracy: %s" % (i,C,A))

# Test model
test_data = {X: mnist.test.images, Y_: mnist.test.labels}

Ctest,Atest = session.run([Cross_Entropy, Accuracy], feed_dict=train_data)
print ("Loss (test): %s, Accuracy (test): %s" % (Ctest,Atest))

    
