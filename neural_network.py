inputs = [1.3, 2.5, 3.6, 1] # shape = (1, 4)

# No. of weights = number of equations (number of equations = number of neurons in the next layer)
weights1 = [1, 1.5, 0.3, 1.2] # shape = (1, 4) (shape of the weights matrix is equal to the number of outputs from the previous layer)
weights2 = [1.3, 0.5, 2.3, -1.2]
weights3 = [2.3, -0.05, 2.3, -0.2]

bias1 = 3 # Only 1 bias per neuron
bias2 = 1
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)