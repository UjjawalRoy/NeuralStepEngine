import numpy as np

# creating an input layer with 4 inputs or batch_size=4, and each input containing 4 features
# shape of which will be (5, 4)
# input_shape = (5, 4)
inputs = [[1.3, 2.5, 3.6, 1],
          [2.3, 5.6, 1.2, 6],
          [3.2, 5, 4.9, 1.5],
          [1.5, 5, 3.2, 4.5],
          [3.6, 1.9, 2, 5]]

# shape of the weights matrix = (3, 4)
weights = [[-1, 1.5, -0.3, 1.2],
           [1.3, 0.5, 2.3, -1.2],      
           [2.3, -0.05, 2.3, -0.2]]

biases = [3, 1, 0.5]                    # shape(biases) = shape(weights)[0] -> (3, )

# to calculate dot product, we need (5, 4).(4, 3), where (4, 3) is Transpose(3, 4)
output = np.dot(inputs, np.array(weights).T) + biases       # putting inputs first since matrix dot product needs (x, y).(y, z) shaped matrices
print(output)