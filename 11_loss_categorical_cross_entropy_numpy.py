import numpy as np

softmax_outputs = np.array([[0.7, 1.0, 2.0],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]])
softmax_outputs_clipped = np.clip(softmax_outputs, 1e-7, 1-1e-7) # clipping all the values by a very small number to avoid encountering log(0.0) anywhere

target_outputs = [0, 1, 1]           # In the last file we saw that one-hot encoding is not necessary, we just need the correct index to locate the predicted vlaue in the softmax_outputs array.
# each value in the target output array is the index that we need to calculate the loss.

#since softmax_outputs is a numpy object, we can us the following code to locate the target predicted value ->
target_predicted_value = softmax_outputs_clipped[[0, 1, 2], target_outputs] #Where, the [0, 1, 2] slices the array along the first dimension (row) and `target_outputs` index wise slices the array along the second dimension in each row.
print(target_predicted_value)

#This can be further generalized as
target_predicted_value = softmax_outputs_clipped[range(len(softmax_outputs_clipped)), target_outputs]
print(target_predicted_value)

# Therefore, the loss can be calculated as ->
loss = -(np.log(softmax_outputs_clipped[range(len(softmax_outputs_clipped)), target_outputs]))
print(loss)

final_loss = np.mean(loss)
print(final_loss)