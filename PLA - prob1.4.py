# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:53:52 2017

@author: Carter

Problem 1.4 from "Learning from Data"

Implementing a simple Perceptron Learning Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt

#tune these params as needed
pointAmount = 50
globalIterations = 5
learningRate = .01


class line(object):
    
    def __init__(self, slope, yintercept):
        self.slope = slope
        self.yintercept = yintercept
        
    def findyval(self, xval):
        return xval*self.slope + self.yintercept


def genPosPoint(line):
    x = np.random.random()*20
    if (np.random.random() < 0.5):
        x = -x
    y = np.random.random()*20
    if (np.random.random() < 0.5):
        y = -y
    
    stuckNum = 0
    
    while (not(y > line.findyval(x))):
        stuckNum = stuckNum + 1
        
        
        y = np.random.random()*100
        if (np.random.random() < 0.5):
            y = -y
    
    return (x,y)

    
def genNegPoint(line):
    x = np.random.random()*20
    if (np.random.random() < 0.5):
        x = -x
    y = np.random.random()*20
    if (np.random.random() < 0.5):
        y = -y
        
    stuckNum = 0
    while (not(y < line.findyval(x))):
        
        
        stuckNum = stuckNum + 1
        
        y = np.random.random()*100
        if (np.random.random() < 0.5):
            y = -y

    return (x, y)



def predict(xvec, weights):
    neuronFires = 0
    activation = weights[0]
    for i in range(xvec.size-1):
        activation = activation + weights[i+1]*xvec[i]
        
    if activation > 0:
        neuronFires = 1
        
    return neuronFires

def train(xvec, lRate, iterations, actual):
    
    weights = [0, 0]
    

    for iteration in range(iterations):
        for j in range(len(xvec)):
            x = xvec[j]
            prediction = predict(x, weights)
            error = actual[j] - prediction
            weights[0] = weights[0] + lRate*error
            weights[1] = weights[1] + lRate*error*x
           


            
    return weights


def PLA():

    xarr = []
    xarr.append(1)
    yarr = []
    yarr.append(0)
    
    for point in points:
        xarr.append(point[0])
        yarr.append(point[1])
    xvec = np.array(xarr)    
    
    signvec = np.empty_like(xvec)
    for i in range(signvec.size):
        if (i%2 == 0):
            signvec[i] = 0
        else:
            signvec[i] = 1
    
    
    weights = train(xvec, learningRate, globalIterations, signvec)
    
    print("Weights: ", weights)
    
    
    g = line(weights[1],weights[0])
    
    return g


    




slope = np.random.random()*5
if (np.random.random() < 0.5):
    slope = -slope
            
yint = np.random.random()*5
if (np.random.random() < 0.5):
    yint = -yint
        
targFunc = line(slope, yint)

points = []
for i in range(0,(int)(pointAmount/2)):
    posPoint = genPosPoint(targFunc)
    negPoint = genNegPoint(targFunc)
    
    plt.scatter(posPoint[0],posPoint[1], c='blue')
    plt.scatter(negPoint[0],negPoint[1], c='red')
    
    points.append(posPoint)
    points.append(negPoint)  
    
x = np.arange(-20,20,0.1)
y = targFunc.findyval(x)

plt.plot(x,y,c='black')
plt.xlabel('x axis')
plt.ylabel('y axis')


g = PLA()
x = np.arange(-20,20,0.1)
y = g.findyval(x)
plt.plot(x,y,c='green')


g2 = g
g2.slope = -g2.slope
x = np.arange(-20,20,0.1)
y = g2.findyval(x)
#plt.plot(x,y,c='orange')



