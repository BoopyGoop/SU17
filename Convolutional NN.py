# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:32:00 2017
@author: Carter Carlos
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/mnist',one_hot=True)


#ARCHITECTURE:
#INPUT -> CONV -> RELU -> POOL -> CONV -> RELU -> POOL -> FC -> OUTPUT

iterations = 3000

numClasses = 10
inputSize = 784
batchSize = 10

nodesFClayer = 1024

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def deepNN(x):
    xReshape = tf.reshape(x, [-1, 28, 28, 1])
    
    #Conv layer 1
    wConv1 = weight_variable([5, 5, 1, 32])
    bConv1 = bias_variable([32])
    #Relu layer 1
    hConv1 = tf.nn.relu(conv2d(xReshape, wConv1) + bConv1)
    
    #Pool layer 1
    hPool1 = maxPool2x2(hConv1)
    
    #Conv layer 2
    wConv2 = weight_variable([5, 5, 32, 64])
    bConv2 = bias_variable([64])
    #Relu layer 2
    hConv2 = tf.nn.relu(conv2d(hPool1, wConv2) + bConv2)
    
    #Pool layer 2
    hPool2 = maxPool2x2(hConv2)  
    
    #Fully connected layer
    wFC = weight_variable([7*7*64, nodesFClayer])
    bFC = bias_variable([nodesFClayer])
    hPool2Flat = tf.reshape(hPool2, [-1, 7*7*64])
    hFC = tf.nn.relu(tf.matmul(hPool2Flat, wFC) + bFC)
    
    keepProb = tf.placeholder(tf.float32)
    hFCdrop = tf.nn.dropout(hFC, keepProb)
    
    
    #Output layer
    wOut = weight_variable([nodesFClayer, numClasses])
    bOut = bias_variable([numClasses])
    
    yConv = tf.matmul(hFCdrop, wOut) + bOut

    return yConv, keepProb



x = tf.placeholder('float', [None, inputSize])
y = tf.placeholder('float', [None, numClasses])

yConv, keepProb = deepNN(x)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yConv))

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.name_scope("xent"):
    correct_prediction = tf.equal(tf.argmax(yConv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter("/mnist_vis/2", sess.graph)  
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    
    
    merged_summary = tf.summary.merge_all()

    
    
    
    for i in range(iterations):
        batch = mnist.train.next_batch(batchSize)
        
        
        #DOES NOT LIKE THIS
        
        if i %5 == 0:
            s=sess.run(merged_summary, feed_dict= {x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
        
        #################
        
        if i %100 == 0:
            
            trainAcc = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keepProb: 1.0})
            print("Step", i, ", training accuracy", trainAcc)
            
        train_step.run(feed_dict= {x: batch[0], y: batch[1], keepProb: 0.5})
        
    print("")
    print("test accuracy", accuracy.eval(feed_dict= { x: mnist.validation.images, y: mnist.validation.labels, keepProb: 1.0}))
