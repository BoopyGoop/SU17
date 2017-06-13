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
learningRate = 1
dimension = 0


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
    #find activation val with dot product
    
    xvec = x
    xvec[len(xvec)-1] = 1
    
    neuronFires = -1
    activation = np.dot(xvec, weights)
    activation = sum(activation[:])
    print(activation)
    
    if activation > 0:
        neuronFires = 1
        
    return neuronFires



def train(data, lRate):
    
    #generalized to be n dimensional
    weights = np.zeros((len(data[0]),1), dtype = float)
    
    hasErrors = True
    iterNum = 0

    while(hasErrors):
        
        iterNum = iterNum +1
        hasErrors = False
        errorCount = 0
        
        for x in data:
            prediction = predict(x, weights)
            error = x[len(x)-1] * prediction
            #print(error)
            
            if (error < 0):
                #update weights
                
                xvec = x
                xvec[len(xvec)-1] = 1
                weights = weights + lRate*error*xvec
                
                hasErrors = True
                errorCount = errorCount +1
        
        print("Iteration ", iterNum, "| #errors ", errorCount )
        

    print(weights)        
    return weights


def PLA(points):
    
    dimension = len(points[0])
    
    data = np.zeros((pointAmount+1, dimension+1), dtype = float)
    data[0, :] = 1
    
    for i in range(pointAmount):
        data[i+1, :dimension] = points[i]
      
    for i in range(pointAmount+1):
        if (i%2 == 0):
            data[i, dimension] = -1
        else:
            data[i, dimension] = 1
    
    
    weights = train(data, learningRate)
    
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




