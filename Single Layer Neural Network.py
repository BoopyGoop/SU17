# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:32:13 2017

@author: Carter Carlos
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#get data

#number of nodes in the hidden layer
numNodesHL = 500

#TODO temp values
numClasses = 10
inputSize = 256



x = tf.placeholder('float', [None, inputSize])
y = tf.placeholder('float')

def NN_model(data):
    
    hiddenLayer = {'weights': tf.Variable(tf.random_normal([inputSize, numNodesHL])), 'biases': tf.Variable(tf.random_normal([numClasses]))}
    
    outputLayer = {'weights': tf.Variable(tf.random_normal([numNodesHL, numClasses])), 'biases': tf.Variable(tf.random_normal([numClasses]))}
    
    hLayer = tf.add(tf.matmul(data, hiddenLayer['weights']), hiddenLayer['biases'])
    hLayer = tf.nn.relu(hLayer)
    
    output = tf.add(tf.matmul(hLayer, outputLayer['weights']), outputLayer['biases'])
    return output

    


def train(x, labelVec):
    prediction = NN_model(x)
    
    #TODO correct?
    sizeOfDataSet = len(x)
    
    #TODO these lines will need to be changed
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    numEpochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(numEpochs):
            epochLoss = 0
            
            for i in range(sizeOfDataSet):
                #TODO fix
                input_vec = x[i]
                label = labelVec[i]
                
                

    
