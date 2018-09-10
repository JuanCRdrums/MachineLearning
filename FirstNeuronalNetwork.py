import numpy as np
import math

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuronalNetwork:

    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedForward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1,self.weights2))

    def backPropagation(self):
        dWeight2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        dWeight1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += dWeight1
        self.weights2 += dWeight2

if __name__ == '__main__':
    x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nw = NeuronalNetwork(x,y)
    for i in range(15000):
        nw.feedForward()
        nw.backPropagation()
    print nw.output
