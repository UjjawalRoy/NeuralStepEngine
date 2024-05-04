"""Softmax function has been tackled separately since it is a bit more complicated than Sigmoid and ReLU"""


initial_outputs = [0.1, 2, -3.9]

exp = 2.718281828459

expo_output = []

for i in initial_outputs:
    expo_output.append(exp**i)           # Exponentiating each value

print("Exponentiated outputs =====> ", expo_output)

norm_output = []
output_total = sum(expo_output)
for output in expo_output:
    norm_output.append(output/output_total) # Normalizing each value by the total of all the exponentiated values in the output layer

print("Normalized outputs =====> ", norm_output)