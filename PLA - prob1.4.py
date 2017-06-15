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
learningRate = .1


class line(object):
    def __init__(self, slope, yintercept):
        self.slope = slope
        self.yintercept = yintercept
        
    def findyval(self, xval):
        return xval*self.slope + self.yintercept


def genPoint(line):
    label = -1
    if (np.random.random() < 0.5):
        label = 1
      
    x = np.random.random()*20
    if (np.random.random() < 0.5):
        x = -x
    y = np.random.random()*20
    if (np.random.random() < 0.5):
        y = -y
    
    if label == -1:
        while (not(y < line.findyval(x))):       
            y = np.random.random()*100
            if (np.random.random() < 0.5):
                y = -y
    else:
        while (not(y > line.findyval(x))):       
            y = np.random.random()*100
            if (np.random.random() < 0.5):
                y = -y
            
    return [label, x, y, 1]


def predict(x, pweights):
    #find activation val with dot product   
    return np.dot(x[1:], pweights.transpose())


def train(data, lRate):
    weights = np.zeros((len(data[0])-1), dtype = float)
    hasErrors = True
    iterNum = 0
    
    #loops until all points are classified correctly
    while(hasErrors):
        
        iterNum = iterNum +1
        hasErrors = False
        errorCount = 0
        
        for x in points:
            if (x[0]*predict(x, weights) <= 0): #if point is misclassified...
                #update weights
                weights = weights + lRate*x[0]*x[1:]
            
                hasErrors = True
                errorCount = errorCount +1
                
        print("Iteration ", iterNum, "| #errors ", errorCount )
 
    return weights


def PLA(points):
    #train the weights
    weights = train(points, learningRate)
    
    print("Weights: ", weights)
    return weights


#randomly generates target function        
slope = np.random.random()*5
if (np.random.random() < 0.5):
    slope = -slope           
yint = np.random.random()*5
if (np.random.random() < 0.5):
    yint = -yint
targFunc = line(slope, yint)

#generates equal number of +1 and -1 classified points
points = np.zeros(shape=(pointAmount, len(genPoint(targFunc))))
for i in range(0, pointAmount):
    point = genPoint(targFunc)
    
    if point[0] == 1:
        color = 'blue'
    else:
        color = 'red'
              
    #TODO this could be generalized to handle more than 2D points?
    plt.scatter(point[1], point[2], c=color)
    points[i] = point
  
#plot target function
x = np.arange(-20,20,0.1)
y = targFunc.findyval(x)
plt.plot(x,y,c='black', label = "Target Function")
plt.xlabel('x axis')
plt.ylabel('y axis')

#use points to generate weights
learnedWeights = PLA(points)

#plot learned function
gSlope = -(learnedWeights[0]/learnedWeights[1])
gInt = -(learnedWeights[2]/learnedWeights[1])
g = line(gSlope, gInt)
x = np.arange(-20,20,0.1)
y = g.findyval(x)
plt.plot(x,y,c='green', label = "Learned Function")

plt.legend(loc='upper left')