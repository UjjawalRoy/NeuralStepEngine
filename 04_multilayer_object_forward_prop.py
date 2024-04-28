import numpy as np

np.random.seed(0)

# ============================================ INPUT ==========================================================================
# creating an input layer with batch_size=5, and each input containing 4 features
# shape of which will be (5, 4)
# input_shape = (5, 4)
X = np.array([[1.3, 2.5, 3.6, 1],
          [2.3, 5.6, 1.2, 6],
          [3.2, 5, 4.9, 1.5],
          [1.5, 5, 3.2, 4.5],
          [3.6, 1.9, 2, 5]])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):   # n_inputs is the number of feature inputs
        # the shape is input-size by the number of neurons to avoid calculating Transpose during feed forward operations
        self.weights = 0.10*(np.random.randn(n_inputs, n_neurons))  #0.10 is used to scaled values from (-1, 1)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)      # number of feature inputs = 4 and number of neurons in current layer = 5
layer2 = Layer_Dense(5, 3)      # number of feature inputs = number of neurons of prev layer = 5 and number of neurons in current layer = 3

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)