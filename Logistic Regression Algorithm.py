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
iterations = 3
learningRate = .1


positiveLabel = 5

def computeGradient(data, weights, labelVec):
    dimension = len(data[0])-1
    sampleSize = len(data)
    innerSum = 0

    #summation
    for i in range(sampleSize):
        x = data[i][1:]
        
        denominator = 1 + np.exp(labelVec[i]*np.dot(weights.transpose(), x))
        
        numerator = np.dot(labelVec[i], x)
        innerSum = innerSum +(numerator/denominator)
    
    gradient = -(innerSum/dimension)
    return gradient


def sigFunction(weights, x):
    return 1/(1+np.exp(-np.dot(weights.transpose(), x)))
    
    
def predict(x, pweights):
    #find activation val with dot product
    return np.dot(x[1:], pweights.transpose())

    

#take construction of label vec outside method
def train(data, lRate, labelVec):
    #initialize weights to 0
    dimension = len(data[0])-1
    weights = np.zeros(dimension, dtype = float)
     
    for j in range(iterations):
                        
        print("Running iteration ", j)
        
        for x in data:
            #compute gradient
            gradient = computeGradient(data, weights, labelVec)
            direction = -gradient
                                
            #update weights
            weights = weights + lRate*direction
    
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

def makeLabelVec(data, positiveLabel):
    #construct label vector
    sampleSize = len(data)
    labelVec = np.zeros(sampleSize, dtype = int)
    for k in range(sampleSize):
        if (data[k][0] == positiveLabel):
            labelVec[k] = 1
        else:
            labelVec[k] = -1    
    return labelVec
                    
    
    

#TRAINING DATA
trainData = np.loadtxt(trainFile)
print("Training Data from: ", trainFile)
print("Dimension: ", len(trainData[0])-1)
print("# of datapoints: ", len(trainData))
labelVec = makeLabelVec(trainData, positiveLabel)

weights = train(trainData, learningRate, labelVec)
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
    titleString = "Image " + str(imageNum) + " labeled as: " + str(label)
    plt.title(titleString)
    
    
    
#investigate points with high error
for i in range(len(lookPoints)):
    plotImage(lookPoints[i])


