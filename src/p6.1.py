import numpy as np

np.random.seed(0)

layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

# ? Softmax Activation Function
# ? Softmax is used for classification problems

# Calculate the exponential values for each output
exp_values = np.exp(layer_outputs)

# Normalize values
norm_base = np.sum(exp_values, axis=1, keepdims=True)
norm_values = exp_values / norm_base

print(norm_values)

# print('Normalized exp values:')
# print(norm_values, end="\n\n")
# print('Sum of normalized values:')
# print(sum(norm_values))
