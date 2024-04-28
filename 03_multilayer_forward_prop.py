import numpy as np

# ============================================ INPUT ==========================================================================
# creating an input layer with 5 inputs or batch_size=5, and each input containing 4 features
# shape of which will be (5, 4)
# input_shape = (5, 4)
inputs = np.array([[1.3, 2.5, 3.6, 1],
          [2.3, 5.6, 1.2, 6],
          [3.2, 5, 4.9, 1.5],
          [1.5, 5, 3.2, 4.5],
          [3.6, 1.9, 2, 5]])

# ============================================ LAYER 1 ==========================================================================
# shape of the weights matrix = (3(number of neurons), 4(number of features in each observation))
weights1 = np.array([[-1, 1.5, -0.3, 1.2],
           [1.3, 0.5, 2.3, -1.2],      
           [2.3, -0.05, 2.3, -0.2]])

biases1 = np.array([3, 1, 0.5])

# ============================================ LAYER 2 ==========================================================================
# shape of the weights matrix = (4(number of neurons), 3(number of outputs per batch from previous layer)) -> no. of neurons in the next layer = 4 and shape of ouput from the previous layer = (5, 3) 
weights2 = np.array([[-0.3, -1.5, 1.3],
           [0.5, 0.9, -2.3],      
           [-2.3, 0.8, -1.4],
           [1.6, 3.2, -0.9]])

biases2 = np.array([2, 0.9, 6, 2])        # shape(biases) = shape(weights)[0] -> (4, )



# ============================================ FORWARD PROPOGATION - INPUT -> LAYER 1 -> LAYER 2 ==================================

print('Input shape = ', inputs.shape)
print('Layer 1 weights shape = ', weights1.shape)
# to calculate dot product, we need (5, 4).(4, 3), where (4, 3) is Trasnspose(3, 4)
layer1_output = np.dot(inputs, weights1.T) + biases1       # putting weights first since matrix dot product needs (x, y).(y, z) shaped matrices
print('Layer 1 output shape = ', layer1_output.shape)
print(layer1_output, '\n')
print('Layer 2 input shape', layer1_output.shape)
print('Layer 2 weights shape', weights2.shape)
# to calculate dot product, we need (5, 3).(3, 4), where (3, 4) is Trasnspose(4, 3)
layer2_output = np.dot(layer1_output, weights2.T) + biases2 
print('Layer 2 output shape = ', layer2_output.shape)
print(layer2_output, '\n')
print('Layer 3 input shape', layer2_output.shape)
