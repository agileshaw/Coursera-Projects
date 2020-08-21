import numpy as np

print('----------------Recap----------------')
weights = np.around(np.random.uniform(size=6), decimals=2);
biases = np.around(np.random.uniform(size=3), decimals=2)
print(weights)
print(biases)

x_1 = 0.5
x_2 = 0.85
print('x1 is {} and x2 is {}'.format(x_1, x_2))

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weigted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The weigted sum of the inputs at the second node in the hidden layer is {}'.format(z_12))

a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(a_11))

a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the second node in the hidden layer is {}'.format(a_12))

#Special attention: the inputs for this layer is a_11 and a_12, not z_11 and z_12
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(z_2))

a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The activation of the node in the output layer is {}'.format(a_2))

results = 1.0/(1.0+np.exp(-0.5825))
print(results)


