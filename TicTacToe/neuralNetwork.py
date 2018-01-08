# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:17:30 2018

@author: dulte
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from copy import copy
from numba import jit,njit



class NeuralNetwork:
    def __init__(self,featureSize,hiddenLayerSize,outputSize,
                 activationFunctionClass="sigmoid",
                 errorFunction="euclidDistance",bias=False,
                 activationFunctionDot="",errorFunctionDot=""):

        self.featureSize = featureSize
        self.hiddeLayerSize = hiddenLayerSize
        self.outputSize = outputSize
        self.perceptron = False
        self.biasNumber = int(bias)


        if isinstance(activationFunctionClass,str):
            if activationFunctionClass == "sigmoid":
                self.activFunc = self.sigmoid
                self.activFuncDot = self.sigmoidDot
            elif activationFunctionClass == "tanh":
                self.activFunc = self.tanh
                self.activFuncDot = self.tanhDot
            elif activationFunctionClass == "identity":
                self.activFunc = self.identity
                self.activFuncDot = self.identityDot
            elif activationFunctionClass == "floor":
                self.activFunc = self.floor
            elif activationFunctionClass == "ceil":
                self.activFunc = self.ceil

            else:
                print("Got unknown activaton function, I am using the \
                          default sigmoid function.")
                self.activFunc = self.sigmoid
                self.activFuncDot = self.sigmoidDot
        elif callable(activationFunctionClass) and callable(activationFunctionDot):
            self.activFunc = activationFunctionClass
            self.activFuncDot = activationFunctionDot
        else:
            print("The given activation function is neither a function\
                      nor a string")
            sys.exit()

        if isinstance(errorFunction,str):
            if errorFunction == "euclidDistance":
                self.errorFunc = self.euclidDistance
                self.errorFuncDot = self.euclidDistanceDot
            elif errorFunction == "difference":
                self.errorFunc = self.difference
                self.errorFuncDot = self.differenceDot
            else:
                print("Got unknown error function, I am using the \
                          default euclidian distance.")
        elif callable(errorFunction) and callable(errorFunctionDot):
            self.errorFunc = errorFunction
            self.errorFuncDot = errorFunctionDot
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

        self.makeNodes()


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


    def makeNodes(self):
        nodes = []

        for i in self.weights:
            nodes.append(np.zeros(i.shape[0]))

        nodes.append(np.zeros(self.outputSize))

        self.nodes = nodes




    def forwardPropagate(self,feature):
        #nextNodes = self.activFunc(np.dot(self.weights[0].T,feature) + self.bias[0])
        nextNodes = self.activFunc(np.dot(feature,self.weights[0]) + self.bias[0])
        self.nodes[0] = np.array(feature)

        for i in range(1,len(self.weights)):
            self.nodes[i] = nextNodes
            nextNodes = self.activFunc(np.dot(nextNodes,self.weights[i])+ self.bias[i])

        self.nodes[-1] = nextNodes
        return nextNodes



    def backPropagate(self,output,real):
        nodeError = copy(self.weights)
        reversedNodes = self.nodes[:]
        reversedNodes.reverse()

        d_L = self.errorFuncDot(output,real)*self.activFuncDot(self.nodes[-1])
        nodeError[0] = np.dot(reversedNodes[1].T,d_L)
        for i in range(1,len(self.weights)):
            nodeError[i] = np.dot(reversedNodes[i+1].T,d_L)*\
                            self.activFuncDot(reversedNodes[i])

        print(nodeError)

        for i, error in enumerate(nodeError):
            if not isinstance(error,np.ndarray):
                self.weights[i][0] -= error
            else:
                print(self.weights[i],np.array(error))
                self.weights[i] -= np.array(error)




    def makeAndGate(self):
        self.featureSize = 2
        self.ourputSize = 1
        self.weights, self.bias = self.makePerceptron()
        self.makeNodes()
        self.weights[0] = np.array([0.6,.6])
        self.activFunc = self.floor


    def makeXorGate(self):
        self.featureSize = 2
        self.ourputSize = 1
        self.hiddeLayerSize = (1,2)
        self.weights,self.bias = self.makeWeightsFromTuple()
        self.makeNodes()
        self.weights[0] = np.array([[.6,1.1],[.6,1.1]])
        self.weights[1] = np.array([-2,1.1])

        self.activFunc = self.floor



    def sigmoid(self,x):
        return 1./(1+np.exp(-x))

    def sigmoidDot(self,x):
        return np.exp(-x)/(1+np.exp(-x))**2

    def tanh(self,x):
        return np.tanh(x)

    def tanhDot(self,x):
        return (1./np.cosh(x))**2

    def identity(self,x):
        return x

    def identityDot(self,x):
        return 1

    def floor(self,x):
        return np.floor(x)

    def ceil(self,x):
        return np.ceil(x)

    def euclidDistance(self,guess,real,axis=0):
        return 1./(2*real.shape[axis])*np.sum((real-guess)**2,axis=axis)

    def euclidDistanceDot(self,guess,real,axis=0):
        return -1./real.shape[axis]*np.sum((real-guess),axis=axis)

    def difference(self,guess,real,axis=0):
        return 1./real.shape[axis]*np.sum((real-guess),axis=axis)

    def differenceDot(self,guess,real,axis=0):
        return -1./real.shape[axis]


def testNNWithANDGate():
    nn = NeuralNetwork(2,(0,0),1)
    nn.makeAndGate()
    features = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [0,0,0,1]
    eps = 1e-6
    print(nn.forwardPropagate(features))
    for f,o in zip(features,outputs):
        assert abs(nn.forwardPropagate(f) - o)<eps, \
                     "Expected {}, but got {}".format(o,nn.forwardPropagate(f))


def testNNWithXORGate():
    nn = NeuralNetwork(2,(1,2),1)
    nn.makeXorGate()
    features = [[0,0],[1,0],[0,1],[1,1]]
    outputs = [0,1,1,0]
    eps = 1e-6
    print(nn.forwardPropagate(features))
    for f,o in zip(features,outputs):
        assert abs(nn.forwardPropagate(f) - o)<eps, \
                     "Expected {}, but got {}".format(o,nn.forwardPropagate(f))

def testBackPropagation():
    nn = NeuralNetwork(1,(0,0),1)
    nn.weights[0] = np.array([1])
    real = np.array([[0]])
    features = [1]
    for i in range(10):
        result = nn.forwardPropagate(features)
        print(result)
        nn.backPropagate(result,real)

    expectedWeights = np.array([0])
    eps = 1e-6
    print(nn.weights)

def testBackPropagationWithXor():
    nn = NeuralNetwork(2,(1,2),1)
    nn.makeXorGate()
    nn.weights[0] = np.array([[.3,4.1],[.1,1.5]])
    nn.weights[1] = np.array([8,-2])

    features = [[0,0],[1,0],[0,1],[1,1]]
    outputs = np.array([0,1,1,0])
    eps = 1e-6

    for i in range(10):
        result = nn.forwardPropagate(features)
        print(result)
        nn.backPropagate(result,outputs)




if __name__=="__main__":
    testNNWithANDGate()
    testNNWithXORGate()
    testBackPropagationWithXor()
