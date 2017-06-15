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
pointAmount = 100
learningRate = .1


class line(object):
    
    def __init__(self, slope, yintercept):
        self.slope = slope
        self.yintercept = yintercept
        
    def findyval(self, xval):
        return xval*self.slope + self.yintercept



#combine point functions
def genPosPoint(line):
    x = np.random.random()*20
    if (np.random.random() < 0.5):
        x = -x
    y = np.random.random()*20
    if (np.random.random() < 0.5):
        y = -y
    while (not(y > line.findyval(x))):
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
    while (not(y < line.findyval(x))):       
        y = np.random.random()*100
        if (np.random.random() < 0.5):
            y = -y
    return (x, y)



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
            
    return [label, x, y]




def predict(x, pweights):
    #find activation val with dot product & sum
    
    xvec = x[:len(x)-1]

    
    #transpose before dot so you don't have to sum
    activation = np.dot(xvec, pweights)
    activation = np.sum(activation)
    
    
    if activation >= 0:
        return 1
    
    return -1 



def train(data, lRate):
    
    #generalized to handle n-dimensional data
    weights = np.zeros((len(data[0])-1), dtype = float)
    hasErrors = True
    iterNum = 0
    
    
    #loops until all points are classified correctly
    while(hasErrors):
        
        iterNum = iterNum +1
        hasErrors = False
        errorCount = 0


        for x in data:
            #predict classification of point based on weights, compare to actual classification
            prediction = predict(x, weights)
            actual = x[len(x)-1]   
            error = actual * prediction
            
            
            
            if (error < 0): #if point is misclassified...
                #update weights
                weights = weights + lRate*actual*x[:len(x)-1]
                

                #graphs the line with the updated weights
                '''
                qSlope = -(weights[0]/weights[1])
                qInt = -(weights[2]/weights[1])
                    
                q = line(qSlope, qInt)
                
                xp = np.arange(-20,20,0.1)
                yp = q.findyval(xp)
                plt.plot(xp,yp,c='red')
                '''
                
                hasErrors = True
                errorCount = errorCount +1
                
                
        print("Iteration ", iterNum, "| #errors ", errorCount )
 
    return weights


def PLA(points):
    dimension = len(points[0])
    
    #constructs data matrix with labels(last column) from points provided
    data = np.zeros((pointAmount, dimension+2), dtype = float) 
    for i in range(1, pointAmount):
        data[i, :dimension] = points[i]
        data[i, dimension] = 1
        
    
    #train the weights
    weights = train(data, learningRate)
    
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
points = []
for i in range(0, pointAmount):
    point = genPoint(targFunc)
    
    if point[0] == 1:
        color = 'blue'
    else:
        color = 'red'
              
        
    #this scatter may not work
    plt.scatter(point[1:len(point)-1],  c=color)
    points.append(point)
  
    
#plot target function
x = np.arange(-20,20,0.1)
y = targFunc.findyval(x)
plt.plot(x,y,c='black')
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
plt.plot(x,y,c='green')




