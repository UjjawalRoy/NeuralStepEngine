inputs = [1.3, 2.5, 3.6, 1] # shape = (4, )

# No. of weights = number of equations (number of equations = number of neurons in the next layer)
# shape = (3, 4) (shape of the weights matrix is equal to the number of neurons in the output layer times outputs from the previous layer/ number of neuron in the previous layer)
weights = [[1, 1.5, 0.3, 1.2],          # ->
           [1.3, 0.5, 2.3, -1.2],       # -> one weight associated with one neuron in the output layer
           [2.3, -0.05, 2.3, -0.2]]     # ->

# Only 1 bias per neuron
biases = [3, 1, 0.5]                    # -> one bias associated with one neuron


# output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
#           inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
#           inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

                                                        # |
                                                        # V

layer_output = []
for neuron_weights, neuron_biases in zip(weights, biases): #loop to create the main equations for the 3 neurons
    neuron_output = 0
    for neuron_input, weight in zip(inputs, neuron_weights): #loop to add the product of individual input to its weight
        neuron_output += neuron_input*weight
    neuron_output += neuron_biases                          #neuron_output will carry the sum of sum(inp*weight) + bias
    layer_output.append(neuron_output)

print(layer_output)