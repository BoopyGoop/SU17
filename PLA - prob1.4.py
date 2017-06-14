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
learningRate = .1
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



def predict(x, pweights):
    #find activation val with dot product
    
    xvec = x[:len(x)-1]
    
    neuronFires = -1
    activation = np.dot(xvec, pweights)
    activation = np.sum(activation)
    
    if activation >= 0:
        neuronFires = 1
    
    print("PREDICTED:::::: ", neuronFires)
    return neuronFires



def train(data, lRate):
    
    #generalized to be n dimensional
    weights = np.zeros((len(data[0])-1), dtype = float)
    hasErrors = True
    iterNum = 0

    #print(data)
    
    while(hasErrors):
        
        iterNum = iterNum +1
        hasErrors = False
        errorCount = 0

        for x in data:
            print(x)
            prediction = predict(x, weights)
            actual = x[len(x)-1]
            print("ACTUAL:::::: ", actual)
            
            error = actual * prediction
            print("ERROR:::::::::: ", error, "\n")
            
            if (error < 0):
                #update weights
                
                print("UPDATING")
                weights = weights + lRate*error*x[:len(x)-1]
                #weights = weights + lRate*prediction*x[:len(x)-1]

                print("New weights----> ", weights, "\n \n")
                
                hasErrors = True
                errorCount = errorCount +1
        
        #print("Iteration ", iterNum, "| #errors ", errorCount )
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
 
    return weights


def PLA(points):
    
    dimension = len(points[0])
    
    
    #constructs data matrix with labels(last column) from points provided
    data = np.zeros((pointAmount, dimension+2), dtype = float) 
    for i in range(pointAmount):
        data[i, :dimension] = points[i]
    for i in range(pointAmount):
        data[i, dimension] = 1
        if (i%2 == 0):
            data[i, dimension+1] = 1
        else:
            data[i, dimension+1] = -1
    
    
    
    #train the weights
    weights = train(data, learningRate)
    
    
    
    print("Weights: ", weights)
    
    return weights




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


learnedWeights = PLA(points)
gSlope = -(learnedWeights[0]/learnedWeights[1])
gInt = -(learnedWeights[2]/learnedWeights[1])

g = line(gSlope, gInt)

x = np.arange(-20,20,0.1)
y = g.findyval(x)
plt.plot(x,y,c='green')




