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
pointAmount = 4
globalIterations = 100
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



def predict(x, weights):
    neuronFires = 0
    activation = weights[0] + weights[1]*x
    
    if activation > 0:
        neuronFires = 1
        
    return neuronFires

def train(xvec, lRate, iterations, actual):
    
    weights = [0, 0]
    
    hasErrors = True
    iterNum = 0

    for iteration in range(iterations):
    #while(hasErrors): 
        iterNum = iterNum +1
        hasErrors = False
        
        errorCount = 0
        for j in range(1, len(xvec)):
            x = xvec[j]
            prediction = predict(x, weights)
            error = actual[j] - prediction
            
            weights[0] = weights[0] + lRate*error
            weights[1] = weights[1] + lRate*error*x
            
            
            if (error != 0):
                hasErrors = True
                errorCount = errorCount +1
        
        print("Iteration ", iteration, "| #errors ", errorCount )
        
        y = line(weights[1], weights[0])
        xp = np.arange(-20,20,0.1)
        yp = y.findyval(xp)
        plt.plot(xp,yp,c='green')

            
    return weights


def PLA(points):

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


g = PLA(points)

x = np.arange(-20,20,0.1)
y = g.findyval(x)
plt.plot(x,y,c='green')




