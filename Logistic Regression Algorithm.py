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

#modify these params to be dynamically decided?
iterations = 5
learningRate = .1

positiveLabel = 5

class line(object):
    def __init__(self, slope, yintercept):
        self.slope = slope
        self.yintercept = yintercept
        
    def findyval(self, xval):
        return xval*self.slope + self.yintercept


def computeGradient(data, weights, labelVec):
    sampleSize = len(data)
    innerSum = 0

    #summation
    for i in range(sampleSize):
        x = data[i][1:]
        
        denominator = 1 + np.exp(labelVec[i]*np.dot(weights.transpose(), x))
        
        numerator = np.dot(labelVec[i], x)
        innerSum = innerSum +(numerator/denominator)
    gradient = -(innerSum/sampleSize)
    return gradient


def sigFunction(weights, x):
    return 1/(1+np.exp(-np.dot(weights.transpose(), x)))
    
    
def predict(x, pweights):
    #find activation val with dot product
    return np.dot(x[1:], pweights.transpose())

    

def train(data, lRate, labelVec):
    #initialize weights to 0
    dimension = len(data[0])-1
    weights = np.zeros(dimension, dtype = float)
     
    for j in range(iterations):
                        
        print("Running iteration", j, "of", iterations-1)
        
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
          

#calculates HORIZONTAL symmetry and intensity
def featureExtract(x):
    x = np.reshape(x, (16,16))   
    #flips horizontally...
    flipx = np.flip(x, 1)
    diff = np.abs(x - flipx)
    sym = -np.sum(np.sum(diff))/256  
    intense = np.sum(np.sum(x))/256
    return [sym, intense]

def createFeatMatrix(data):
    featMatrix = np.zeros((len(data), 4))
    
    for i in range(len(data)):
        extractedFeatures = featureExtract(data[i][1:])
        featMatrix[i][0] = data[i][0]
        featMatrix[i][1] = extractedFeatures[0]
        featMatrix[i][2] = extractedFeatures[1]
        featMatrix[i][3] = 1
    return featMatrix


def calculateEin(featMatrix, weights, labelVec):
    innerSum = 0
    for i in range(len(featMatrix)):
        x = featMatrix[i][1:]
        
        innerSum = innerSum + np.log(1+np.exp(-labelVec[i]*np.dot(weights.transpose(),x)))
        
    return (innerSum/len(featMatrix))
    
def plotPoints(data, positiveLabel):
    
    for x in data:
        if (x[0] == positiveLabel):
            color = 'blue'
        else:
            color = 'red'
        plt.scatter(x[1], x[2], c=color)




def createThirdOrderMatrix(data):
    thirdOrderMatrix = np.zeros((len(data), 11))
    
    for i in range(len(data)):
        x = data[i]
        
        p1 = x[1]
        p2 = x[2]
        
        thirdOrderMatrix[i][0] = x[0]
        thirdOrderMatrix[i][1] = p1
        thirdOrderMatrix[i][2] = p2
        thirdOrderMatrix[i][3] = p1*p1
        thirdOrderMatrix[i][4] = p1*p2
        thirdOrderMatrix[i][5] = p2*p2
        thirdOrderMatrix[i][6] = p1*p1*p1
        thirdOrderMatrix[i][7] = p1*p1*p2
        thirdOrderMatrix[i][8] = p1*p2*p2
        thirdOrderMatrix[i][9] = p2*p2*p2
        thirdOrderMatrix[i][10] = 1

    
    return thirdOrderMatrix




    

                                    #TRAINING DATA
#first order calculation
trainData = np.loadtxt(trainFile)
print("Training Data from: ", trainFile)
print("Dimension: ", len(trainData[0])-1)
print("# of datapoints: ", len(trainData))
print("")
labelVec = makeLabelVec(trainData, positiveLabel)
trainFeatMatrix = createFeatMatrix(trainData)

#save to text file
text_file = open('trainOutput.txt', 'w')
for i in range(len(trainFeatMatrix)):
    text_file.write(str(trainFeatMatrix[i]))
    text_file.write("\n")
text_file.close()

print("training linear...")
weights = train(trainFeatMatrix, learningRate, labelVec)
ein = calculateEin(trainFeatMatrix, weights, labelVec)

#plot training data
plt.xlabel('symmetry')
plt.ylabel('intensity')
plotPoints(trainFeatMatrix, positiveLabel)
trainLine = line(-(weights[0]/weights[1]),-(weights[2]/weights[1]))
x = np.arange(-0.5, 0, 0.01)
y = trainLine.findyval(x)
plt.plot(x,y,c='black')
plt.title("linear training data, Ein= " + str(ein))



#third order calculation
print("\n")
print("training third order...")
thirdOrderMatrix = createThirdOrderMatrix(trainFeatMatrix)
thirdOrderWeights = train(thirdOrderMatrix, learningRate, labelVec)

einThirdOrder = calculateEin(thirdOrderMatrix, thirdOrderWeights, labelVec)




print("\n")




                                #TESTING DATA
#first order calculation
testData = np.loadtxt(testFile)
print("Testing Data from: ", testFile)
print("Dimension: ", len(testData[0])-1)
print("# of datapoints: ", len(testData))
testFeatMatrix = createFeatMatrix(testData)

#save to text file
text_file = open('testOutput.txt', 'w')
for i in range(len(testFeatMatrix)):
    text_file.write(str(testFeatMatrix[i]))
    text_file.write("\n")
text_file.close()

#errorVec = test(testFeatMatrix, weights, positiveLabel)
#print("Errors: ", errorVec)

labelVecTest = makeLabelVec(testData, positiveLabel)
einTest = calculateEin(testFeatMatrix, weights, labelVecTest)

#plot testing data
plt.figure()
plt.xlabel('symmetry')
plt.ylabel('intensity')
plotPoints(testFeatMatrix, positiveLabel)
testLine = line(-(weights[0]/weights[1]),-(weights[2]/weights[1]))

x = np.arange(-0.5, 0, 0.01)
y = testLine.findyval(x)
plt.plot(x,y,c='black')

plt.title("linear testing data, Ein= " + str(einTest))


#third order calculation
thirdOrderMatrix = createThirdOrderMatrix(testFeatMatrix)
einTestThirdOrder = calculateEin(thirdOrderMatrix, thirdOrderWeights, labelVecTest)




                                #PRINT EACH Ein
print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Linear Training Ein=", ein)
print("Linear Testing Ein=", einTest)
print("")
print("Third Order Training Ein=", einThirdOrder)
print("Third Order Testing Ein=", einTestThirdOrder)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")




'''
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
'''

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
    titleString = "Image " + str(imageNum) + " labeled as: " + "\"" + str(label) + "\""
    plt.title(titleString)
    
    
'''   
#investigate points with high error
for i in range(len(lookPoints)):
    plotImage(lookPoints[i])
'''
