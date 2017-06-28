# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:32:13 2017

implementation of a single layer Neural Network, tested on mnist data set

@author: Carter Carlos
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/mnist',one_hot=True)

#number of nodes in the hidden layer
numNodesHL = 100

iterations = 10


numClasses = 10
inputSize = 784
batchSize = 10


x = tf.placeholder('float', [None, inputSize])
y = tf.placeholder('float')

def NN_model(data):
    
    #dictionaries hold wiehgts/biases of each layer, randomly generated at start
    hiddenLayer = {'weights': tf.Variable(tf.random_normal([inputSize, numNodesHL])), 'biases': tf.Variable(tf.random_normal([numNodesHL]))}
    outputLayer = {'weights': tf.Variable(tf.random_normal([numNodesHL, numClasses])), 'biases': tf.Variable(tf.random_normal([numClasses]))}
    
    hLayer = tf.add(tf.matmul(data, hiddenLayer['weights']), hiddenLayer['biases'])
    hLayer = tf.nn.relu(hLayer)
    output = tf.add(tf.matmul(hLayer, outputLayer['weights']), outputLayer['biases'])
    
    return output

    


def train(x, iterations):
    prediction = NN_model(x)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for iteration in range(iterations):
            iterationLoss = 0
            
            for i in range(int(mnist.train.num_examples/batchSize)):
                xIter, yIter = mnist.train.next_batch(batchSize)
                i, c = sess.run([optimizer, cost], feed_dict = {x: xIter, y: yIter})
                iterationLoss = iterationLoss + c

            print("Iteration", iteration, "of", iterations-1, "completed")
            print("Loss:", iterationLoss)
            print("")

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy:", accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels}))


train(x, iterations)
