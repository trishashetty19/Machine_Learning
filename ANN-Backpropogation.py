import numpy as np 

X=np.array(([2,9],[1,5],[3,6]),dtype=float)
Y=np.array(([95],[86],[70]),dtype=float)

X=X/np.amax(X,axis=0)
Y=Y/100

print(X)
print(Y)

class NeuralNetwork(object):
    def __init__(self):
        self.inputSize=2
        self.outputSize=1
        self.hiddenSize=3

        self.W1=np.random.randn(self.inputSize,self.hiddenSize)
        self.W2=np.random.randn(self.hiddenSize,self.outputSize)


    def forward(self,X):
        self.z=np.dot(X,self.W1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.W2)
        o=self.sigmoid(self.z3)
        return o 
    
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
    def sigmoidPrime(self,s):
        return s*(s-1)

    def backward(self,X,Y,o):
        self.o_error=Y-o
        self.o_delta=self.o_error*self.sigmoidPrime(o)
        self.z2_error=self.o_delta.dot(self.W2.T)
        self.z2_delta=self.z2_error*self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self,X,Y):
        o=self.forward(X)
        self.backward(X,Y,o)

NN=NeuralNetwork()
for i in range (1):
        print("Input"+str(X))
        print("Actual Output"+str(Y))
        print("Predicted output"+str(NN.forward(X)))
        print("Loss"+str(np.mean(np.square(Y-NN.forward(X)))))

        NN.train(X,Y)