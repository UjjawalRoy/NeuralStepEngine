"""
The formula for calculating categorical cross entropy is 
                Loss = -∑(Yi*log(Yi_pred))               (https://365datascience.com/tutorials/machine-learning-tutorials/cross-entropy-loss/)
where,
 Yi is the actual class value for the observation after one-hot encoding (https://medium.com/@irvan.rahadhian/one-hot-encoding-what-and-why-f22d11a7602a)
 Yi_pred is teh predicted probablity value for the same observation
 The `log` is considered to be a natural log (to the base `e=2.7182...`)
"""

import math

softmax_output = [0.7, 0.2, 0.1]

target_output = [1.0, 0.0, 0.0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])
print(loss)

"""
since the other target values are 0, their product will also be zero and
the target value of our target class is 1, the multiplocation essentially returns
The formula - 
                loss = -log(Yi_pred)
where,
 i = the index of the target class
"""

loss = -math.log(softmax_output[0])
print(loss)