# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:17:30 2018

@author: dulte
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from numba import jit,njit



class NeuralNetwork:
    def __init__(self,featureSize,hiddenLayerSize,outputSize,
                 activationFunctionClass="sigmoid",
                 errorFunction="euclidDistance",bias=False):
        
        self.featureSize = featureSize
        self.hiddeLayerSize = hiddenLayerSize
        self.outputSize = outputSize
        self.perceptron = False
        self.biasNumber = int(bias)
        
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
            self.weights,self.bias = self.makeWeightsFromList()
        elif isinstance(hiddenLayerSize,tuple):
            if hiddenLayerSize[0] == 0 or hiddenLayerSize[1] == 0:
                self.weights,self.bias = self.makePerceptron()
            else:
                self.weights,self.bias = self.makeWeightsFromTuple()
            
        else:
            print("The hidden layer size was given in an unknown way, \
                     it has to be a tuple, or a list or an array")
            
            
            
     
    def makeWeightsFromList(self):
        """
        Size: (input,output)
        """
        weights = []
        bias = []
        weights.append(np.random.normal(size=(self.featureSize,self.hiddeLayerSize[0])))
        bias.append(np.ones(self.hiddeLayerSize[0])*self.biasNumber)
        
        
        for i in range(1,range(self.hiddeLayerSize)):
            weights.append(np.random.normal(size=(self.hiddeLayerSize[i-1],self.hiddeLayerSize[i])))
            bias.append(np.ones(self.hiddeLayerSize[i])*self.biasNumber)
        
        weights.append(np.random.normal(size=(self.hiddeLayerSize[-1],self.outputSize)))
        bias.append(np.ones(self.outputSize)*self.biasNumber)
        
        return weights,bias
    
    def makeWeightsFromTuple(self):
        weights = []
        bias = []
        weights.append(np.random.normal(size=(self.featureSize,self.hiddeLayerSize[1])))
        bias.append(np.ones(self.hiddeLayerSize[0])*self.biasNumber)
        
        for i in range(1,self.hiddeLayerSize[0]):
            
            weights.append(np.random.normal(size=(self.hiddeLayerSize[i-1],self.hiddeLayerSize[i])))
            bias.append(np.ones(self.hiddeLayerSize[i])*self.biasNumber)
       
        weights.append(np.random.normal(size=(self.hiddeLayerSize[1],self.outputSize)))
        bias.append(np.ones(self.outputSize)*self.biasNumber)
        
        return weights,bias
    
    
    def makePerceptron(self):
        self.perceptron = True
        bias = [np.ones(self.outputSize)*self.biasNumber]
        return [np.random.normal(size=(self.featureSize,self.outputSize))],bias
 
    
    def forwardPropagate(self,feature):
        activFunc = njit(self.activFunc)
        bias = self.bias
        weights = self.weights
        
        #@jit(nopython=True)
        def forProg(f):
        
            nextNodes = activFunc(np.dot(weights[0].T,f) + bias[0])
            
            for i in range(1,len(weights)):
                nextNodes = activFunc(np.dot(weights[i].T,nextNodes)+ bias[i])

            return nextNodes
    
        return forProg(feature)
    
    def makeAndGate(self):
        self.featureSize = 2
        self.ourputSize = 1
        self.weights, self.bias = self.makePerceptron()
        self.weights[0] = np.array([0.6,.6])
        self.activFunc = self.floor
        
        
    def makeXorGate(self):
        self.featureSize = 2
        self.ourputSize = 1
        self.hiddeLayerSize = (1,2)
        self.weights,self.bias = self.makeWeightsFromTuple()
        
        
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
    
    
    
    
    
    