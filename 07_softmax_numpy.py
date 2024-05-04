"""Softmax function has been tackled separately since it is a bit more complicated than Sigmoid and ReLU"""

import numpy as np

initial_outputs = [0.1, 2, -3.9]

exp_output = np.exp(initial_outputs)

norm_outputs = exp_output / np.sum(exp_output)

print(norm_outputs)
print(sum(norm_outputs))                     # Must be equal to 1 since this outputs a probability ditribution!!!