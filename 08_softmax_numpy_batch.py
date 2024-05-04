"""Softmax function has been tackled separately since it is a bit more complicated than Sigmoid and ReLU"""

import numpy as np

# creating an output layer with batch_size=3, and each output containing 3 features
# shape of which will be (3, 3)
# output_shape = (3, 3)
initial_outputs = [[0.1, 2, -3.9],
                   [-2, -0.6, 0.9],
                   [3.5, 8.3, 1.2]]

offset_val = np.max(initial_outputs, axis=1, keepdims=True)
print("The offset values =====> ", offset_val)
max_val_offset = np.subtract(initial_outputs, offset_val)
print("The initial outputs after max value offset =====> ", max_val_offset)

"""All the values in each batch is offsetted by the max value to
avoid explosion of values if the power of an exponent is very large
example: e^(100000) ----> may result in memory overflow error"""

exp_output = np.exp(initial_outputs)         # Line of code remain untouched since np.exp() works on all individual values of the 2D matrix
normalization_base = np.sum(exp_output, axis=1, keepdims=True)

"""axis=1 enables us to add the values along the row in a 2D matrix (axis=0 would do the same along the column);
keepdims=True enables us to create a matrix with the shape of the initial_outputs shape. So, instead of getting an output of shape=(3,)
we will get an output of shape=(3, 1). This further helps with row wise division of the 2 matrices."""

print("The normalized base =====> ", normalization_base)
print("Shape of normalized base = ", normalization_base.shape)

norm_outputs = exp_output / normalization_base

print(norm_outputs)
print(np.sum(norm_outputs, axis=1, keepdims=True))                     # Must be equal to 1 since this outputs a probability ditribution!!!