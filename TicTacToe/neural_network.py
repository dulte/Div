import matplotlib.pyplot as plt
import numpy as np
import sys


class NeuralNetwork:
    def __init__(self,layer_size_list):
        self.layer_size_list = layer_size_list
        self.number_of_layers = len(layer_size_list)

        self.make_weights_and_biases()
        self.make_nodes_and_activations()

        self.activation_function = self.sigmoid
        self.activation_function_prime = self.sigmoid_prime



    def make_weights_and_biases(self):
        self.weights = [np.random.normal(size=(x,y)) \
            for x,y in zip(self.layer_size_list[:-1],self.layer_size_list[1:])]
        self.biases = [np.random.normal(size=x) for x in self.layer_size_list[1:]]

    def make_nodes_and_activations(self):
        self.nodes = [np.zeros(x) for x in self.layer_size_list[1:]]
        self.activations = [np.zeros(x) for x in self.layer_size_list]

    def feed_forward(self,feature):
        feature = np.array(feature)

        activation = feature
        self.activations[0] = feature

        for i in range(len(self.weights)):
            print(self.weights[i])
            node = np.dot(self.weights[i],activation)# + self.biases[i]
            print(node)
            print("hei")
            self.nodes[i] = node

            activation = self.activation_function(node)
            self.activations[i+1] = activation

        return activation

    def feed_back(self,output,real):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        delta = self.cost_prime(output,real)*self.activation_function_prime(self.nodes[-1])
        delta = delta.reshape(delta.shape[0],1)
        
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,self.activations[-2].T)
        
        
        
        print(nabla_w[-1].shape,self.weights[-1].shape)

        for l in range(2,self.number_of_layers):
            z = self.nodes[-l]

            delta = np.dot(self.weights[-l+1].T,delta)*\
                    self.activation_function_prime(z)

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,self.activations[-l-1].T)


        return nabla_w,nabla_b

    def update_network(self,nabla_w,nabla_b,eta=0.1):
        self.weights = [w - eta*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b - eta*nb for b,nb in zip(self.biases,nabla_b)]


    def train(self,training_data,epochs,eta=0.1):
        for e in range(epochs):
            inputs = training_data[0]
            outputs = training_data[1]
            for i,o in zip(inputs,outputs):
                i = np.array(i)
                o = np.array(o)

                result = self.feed_forward(i)
                nabla_w,nabla_b = self.feed_back(result,o)

                self.update_network(nabla_w,nabla_b)

            print("Done with {}/{} epochs".format(e,epochs))
            print("Current Cost {}".format(self.cost(result,o)))

        else:
            print("Done With {} Epochs".format(epochs))







    def sigmoid(self,x):
        return 1./(1+np.exp(-x))

    def sigmoid_prime(self,x):
        return np.exp(-x)/(1+np.exp(-x))**2

    def cost_prime(self,output,real):
        return (output-real)

    def cost(self,output,real):
        return (real-output)**2


if __name__ == '__main__':
    nn = NeuralNetwork([2,2,1])
#    inputs = [[[0],[0]],[[1],[1]],[[1],[0]],[[0],[1]]]
    inputs = [[0,0],[1,1],[1,0],[0,1]]
    outputs = [[0],[0],[1],[1]]
    training_data = [inputs,outputs]
    nn.train(training_data,10)
