import numpy as np

inputs = [1.3, 2.5, 3.6, 1]

weights = [[-1, 1.5, -0.3, 1.2],
           [1.3, 0.5, 2.3, -1.2],      
           [2.3, -0.05, 2.3, -0.2]]

biases = [3, 1, 0.5]

output = np.dot(weights, inputs) + biases       # putting weights first since matrix dot product needs (x, y).(y, z) shaped matrices
print(output)