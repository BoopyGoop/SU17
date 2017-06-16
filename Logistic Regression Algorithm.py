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


def computeGradient(data, weights):
    
    
def updateWeights(direction, weights, lRate):
    

def logRegressAlgo(data):
    
    
    










#import training data
data = np.loadtxt(trainFile)
print(data)
print("Cols:", len(data[0]))
print("Rows:", len(data))