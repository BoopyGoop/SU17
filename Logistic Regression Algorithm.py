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

#TODO modify these params to be dynamically decided?
iterations = 10
learningRate = .1


positiveLabel = 5


def computeGradient(data, weights, labelVec):
    dimension = len(data[0])-1
    innerSum = 0
    
    #summation
    for i in range(0, dimension-1):
        x = data[i][1:]
        
        denom = 1 + np.exp(labelVec[i]*np.dot(weights.transpose(), x))
        
        num = np.dot(labelVec[i], x)
        innerSum = innerSum +(num/denom)
    
    gradient = -(innerSum/dimension)
    return gradient


def sigFunction(weights, x):
    return 1/(1+np.exp(-np.dot(weights.transpose(), x)))
    
    
def predict(x, pweights):
    #find activation val with dot product
    return np.dot(x[1:], pweights.transpose())

    
def train(data, lRate, positiveLabel):
    #initialize weights to 0
    dimension = len(data[0])-1
    weights = np.zeros(dimension, dtype = float)
     
    #construct label vector
    labelVec = np.zeros(dimension, dtype = int)
    for k in range(dimension):
        if (data[k,0] == positiveLabel):
            labelVec[k] = 1
        else:
            labelVec[k] = -1                        
    
    for j in range(iterations):

        
        for x in data:                
            if (labelVec[j]*predict(x, weights) <= 0): #if point is misclassified...
                #compute gradient
                gradient = computeGradient(data, weights, labelVec)
                direction = -gradient
                                
                #update weights
                weights = weights + lRate*direction
                
        print("Running iteration ", j)
    
    return weights  
  
    
def test(data, weights, positiveLabel):
    pointAmount = len(data)
    
    percentVec = np.zeros(pointAmount)
    
    for i in range(pointAmount):
        percentVec[i] = sigFunction(weights, data[i][1:])
    
    errorVec = np.zeros(pointAmount)
    for j in range(pointAmount):
        if (data[j][0] == positiveLabel):
            errorVec[j] = 1 - percentVec[j]
        else:
            errorVec[j] = percentVec[j]
    
    return errorVec
    

#TRAINING DATA
trainData = np.loadtxt(trainFile)
print("Training Data from: ", trainFile)
print("Dimension: ", len(trainData[0])-1)
print("# of datapoints: ", len(trainData))
weights = train(trainData, learningRate, positiveLabel)
#print("Weights: ", weights)

print("\n")


#TESTING DATA
testData = np.loadtxt(testFile)
print("Testing Data from: ", testFile)
print("Dimension: ", len(testData[0])-1)
print("# of datapoints: ", len(testData))
errorVec = test(testData, weights, positiveLabel)
#print("Errors: ", errorVec)


#plot the errors

lookPoints = []
xval = 0
for error in errorVec:
    if error>0.5:
        color = 'red'
        lookPoints.append(xval)
    else:
        color = 'blue'
    plt.scatter(xval, error, c=color)
    xval = xval+1

plt.xlabel('data')
plt.ylabel('error')

print (">0.5 error at: ", lookPoints)


def plotImage(imageNumber):
    imageNum = imageNumber
    imageVec = testData[imageNum][1:]
    
    label = testData[imageNum][0]
    if (label == positiveLabel):
        label = 5
    else:
        label = 3
    
    imageMatrix = np.zeros((16,16))
    
    current = 0
    for row in range(16):
         for column in range(16):
           
             imageMatrix[row][column] = imageVec[current]
             current = current +1

    plt.figure()
    plt.imshow(imageMatrix, cmap = 'gray')
    print("IMAGE", imageNum, "WAS LABELED AS:", label)
    
    
    
#investigate points with high error
for i in range(len(lookPoints)):
    plotImage(lookPoints[i])


