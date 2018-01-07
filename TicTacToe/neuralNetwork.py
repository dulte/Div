# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:17:30 2018

@author: dulte
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from numba import jit



class NeuralNetwork:
    def __init__(self,featureSize,hiddenLayerSize,outputSize,
                 activationFunctionClass="sigmoid",
                 errorFunction="euclidDistance"):
        
        self.featureSize = featureSize
        self.hiddeLayerSize = hiddenLayerSize
        self.ourputSize = outputSize
        self.perceptron = False
        
        if isinstance(activationFunctionClass,str):
            if activationFunctionClass == "sigmoid":
                self.activFunc = self.sigmoid
            elif activationFunctionClass == "tanh":
                self.activFunc = self.tanh
            elif activationFunctionClass == "identity":
                self.activFunc = self.identity
            elif activationFunctionClass == "floor":
                self.activFunc = self.floor
            elif activationFunctionClass == "ceil":
                self.activFunc = self.ceil
            
            else:
                print("Got unknown activaton function, I am using the \
                          default sigmoid function.")
                self.activFunc = self.sigmoid
        elif callable(activationFunctionClass):
            self.activFunc = activationFunctionClass
        else:
            print("The given activation function is neither a function\
                      nor a string")
            sys.exit()
            
        if isinstance(errorFunction,str):
            if errorFunction == "euclidDistance":
                self.errorFunc = self.euclidDistance
            elif errorFunction == "difference":
                self.errorFunc = self.difference
            else:
                print("Got unknown error function, I am using the \
                          default euclidian distance.")
        elif callable(errorFunction):
            self.errorFunc = errorFunction
        else:
            print("The given error function is neither a function\
                      nor a string")
            sys.exit()
            
        
        if isinstance(hiddenLayerSize,list) or isinstance(hiddenLayerSize,np.ndarray):
            self.weights = self.makeWeightsFromList()
        elif isinstance(hiddenLayerSize,tuple):
            if hiddenLayerSize[0] == 0 or hiddenLayerSize[1] == 0:
                self.weights = self.makePerceptron()
            else:
                self.weights = self.makeWeightsFromTuple()
            
        else:
            print("The hidden layer size was given in an unknown way, \
                     it has to be a tuple, or a list or an array")
            
            
            
     
    def makeWeightsFromList(self):
        """
        Size: (input,output)
        """
        weights = []
        weights.append(np.random.normal(size=(self.featureSize,self.hiddeLayerSize[0])))
        
        for i in range(1,range(self.hiddeLayerSize)):
            weights.append(np.random.normal(size=(self.hiddeLayerSize[i-1],self.hiddeLayerSize[i])))
        
        weights.append(np.random.normal(size=(self.hiddeLayerSize[-1],self.ourputSize)))
        
        return weights
    
    def makeWeightsFromTuple(self):
        weights = []
        weights.append(np.random.normal(size=(self.featureSize,self.hiddeLayerSize[1])))
        for i in range(1,self.hiddeLayerSize[0]):
            
            weights.append(np.random.normal(size=(self.hiddeLayerSize[i-1],self.hiddeLayerSize[i])))
        
       
        weights.append(np.random.normal(size=(self.hiddeLayerSize[1],self.ourputSize)))
        
        return weights
    
    
    def makePerceptron(self):
        self.perceptron = True
        return [np.random.normal(size=(self.featureSize,self.ourputSize))]
 
    @jit
    def forwardPropagate(self,feature):

        nextNodes = self.activFunc(np.dot(self.weights[0].T,feature))
        
        for i in range(1,len(self.weights)):
            nextNodes = self.activFunc(np.dot(self.weights[i].T,nextNodes))
        
        return nextNodes
    
    def makeAndGate(self):
        self.featureSize = 2
        self.ourputSize = 1
        self.weights = self.makePerceptron()
        self.weights[0] = np.array([0.6,.6])
        self.activFunc = self.floor
        
        
    def makeXorGate(self):
        self.featureSize = 2
        self.ourputSize = 1
        self.hiddeLayerSize = (1,2)
        self.weights = self.makeWeightsFromTuple()
        
        
        self.weights[0] = np.array([[.6,1.1],[.6,1.1]])
        self.weights[1] = np.array([-2,1.1])
        
        self.activFunc = self.floor
    
    
    
    def sigmoid(self,x):
        return 1./(1+np.exp(-x))
    
    def tanh(self,x):
        return np.tanh(x)
    
    def identity(self,x):
        return x
    
    def floor(self,x):
        return np.floor(x)
    
    def ceil(self,x):
        return np.ceil(x)
    
    def euclidDistance(selt,guess,real,axis=0):
        return np.sum((real-guess)**2,axis=axis)
    
    def difference(self,guess,real,axis=0):
        return np.sum((real-guess),axis=axis)
    
    
def testNNWithANDGate():
    nn = NeuralNetwork(2,(0,0),1)
    nn.makeAndGate()
    features = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [0,0,0,1]
    eps = 1e-6
    for f,o in zip(features,outputs):
        assert abs(nn.forwardPropagate(f) - o)<eps, \
                     "Expected {}, but got {}".format(o,nn.forwardPropagate(f))
                     

def testNNWithXORGate():
    nn = NeuralNetwork(2,(1,2),1)
    nn.makeXorGate()
    features = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [0,1,1,0]
    eps = 1e-6
    for f,o in zip(features,outputs):
        assert abs(nn.forwardPropagate(f) - o)<eps, \
                     "Expected {}, but got {}".format(o,nn.forwardPropagate(f))

if __name__=="__main__":
    testNNWithANDGate()
    testNNWithXORGate()
    
    
    
    
    
    