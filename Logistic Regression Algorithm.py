# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:54:37 2017

@author: Carter Carlos


Implementation of Logistic Regression Algorithm to distinguish written numbers 3 and 5
"""

import numpy as np
import matplotlib.pyplot as plt

testFile = "zip.test"
trainFile = "zip.train"

#TODO modify these params to be dynamically decided
iterations = 50
learningRate = 0.1
positiveLabel = 5




def computeGradient(data, weights, labelVec):

    pointAmount = len(data[0])-1
    innerSum = 0
    
    #summation
    for i in range(0, pointAmount-1):
        x = data[i[1:]]
        
        denom = 1 + np.exp(labelVec[i]*np.dot(weights.transpose(), x))
        
        #TODO dot product or just multiply?
        num = np.dot(labelVec[i], x)
        
        #TODO probably will have problems with the divide...
        innerSum = innerSum +(num/denom)
    
    #TODO divide problem?
    gradient = -(innerSum/pointAmount)
    return gradient


def sigFunction(weights, x):
    return 1/(1+np.exp(-np.dot(weights.transpose(), x)))
    

    
def predict(x, pweights):
    #find activation val with dot product
    return np.dot(x[1:], pweights.transpose())

    
def train(data, lRate, positiveLabel):
    #initialize weights to 0
    weights = np.zeros((len(data[0])-1), dtype = float)
    pointAmount = len(data[0])-1
                
    #construct label vector
    for k in range(pointAmount):
        if (data[k,0] == positiveLabel):
            label = 1
        else:
            label = -1                        
    
    for j in range(iterations):
        errorCount = 0
        for x in data:
            
            #TODO just make a seperate label vector
            if (x[0] == positiveLabel):
                label = 1
            else:
                label = -1
                
            
            if (label*predict(x, weights) <= 0): #if point is misclassified...
                #compute gradient
                gradient = computeGradient(data, weights, labelVec)
                direction = -gradient
                
                #update weights
                weights = weights + lRate*direction
                
                
                errorCount = errorCount +1
                
        print("Iteration ", j, "| #errors ", errorCount )
        
        
        
#TODO this function is probably unnecessary...
def logisticRegressionAlgo(data, positiveLabel):
    weights = train(data, learningRate, positiveLabel)

    return weights




#import training data
data = np.loadtxt(trainFile)
print(data)
print("Cols:", len(data[0]))
print("Rows:", len(data))

weights = logisticRegressionAlgo(data, positiveLabel)


