# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:54:37 2017

@author: Carter Carlos


Implementation of Logistic Regression Algorithm to distinguish written numbers 3 and 5
"""

import numpy as np

testFile = "zip.test"
trainFile = "zip.train"

#TODO modify these params to be dynamically decided
iterations = 10
learningRate = .1
positiveLabel = 5




def computeGradient(data, weights, labelVec):

    pointAmount = len(data[0])-1
    innerSum = 0
    
    #summation
    for i in range(0, pointAmount-1):
        x = data[i][1:]
        
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
    pointAmount = len(data[0])-1
    weights = np.zeros(pointAmount, dtype = float)
    
    
    #construct label vector
    labelVec = np.zeros(pointAmount, dtype = int)
    for k in range(pointAmount):
        if (data[k,0] == positiveLabel):
            labelVec[k] = 1
        else:
            labelVec[k] = -1                        
    
    for j in range(iterations):
        errorCount = 0
        
        for x in data:                
            
            if (labelVec[j]*predict(x, weights) <= 0): #if point is misclassified...
                #compute gradient
                gradient = computeGradient(data, weights, labelVec)
                direction = -gradient
                
                #update weights
                weights = weights + lRate*direction
                
                
                errorCount = errorCount +1
                
        print("Iteration ", j, "| #errors ", errorCount )
        
    
    return weights  
  
    
def test(data, weights):
    
    #TODO add a comparison to the labels provided by the test data
    pointAmount = len(data[0])-1
    percentVec = np.zeros(pointAmount)
    
    for i in range(pointAmount):
        percentVec[i] = sigFunction(weights, data[i][1:])
    
    
    return percentVec
    



trainData = np.loadtxt(trainFile)
print("Training Data from: ", trainFile)
print("Dimension: ", len(trainData[0])-1)
print("# of datapoints: ", len(trainData))
weights = train(trainData, learningRate, positiveLabel)
print("Weights: ", weights)

print("\n")

testData = np.loadtxt(trainFile)
print("Testing Data from: ", testFile)
print("Dimension: ", len(testData[0])-1)
print("# of datapoints: ", len(testData))

percentVec = test(testData, weights)

print(percentVec)

