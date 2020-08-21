import numpy as np
from random import seed

print('----------------Initialize a Network----------------')

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_outputs):
	num_nodes_previous = num_inputs
	network = {}

	for layer in range(num_hidden_layers+1):
		if layer == num_hidden_layers:
			layer_name = 'output'
			num_nodes = num_nodes_outputs
		else:
			layer_name = 'layer_{}'.format(layer+1)
			num_nodes = num_nodes_hidden[layer]

		network[layer_name] = {}
		for node in range(num_nodes):
			node_name = 'node_{}'.format(node+1)
			network[layer_name][node_name] = {
				'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
				'bias': np.around(np.random.uniform(size=1), decimals=2),
			}
		num_nodes_previous = num_nodes
	return(network)

def compute_weighted_sum(inputs, weights, bias):
	return np.sum(inputs*weights)+bias

def node_activation(weighted_sum):
	return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

def forward_propagation(network, inputs):
	layer_inputs = list(inputs)

	for layer in network:
		layer_data = network[layer]
		layer_outputs = []
		for node in layer_data:
			node_data = layer_data[node]
			node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
			#why use node_output[0] instead of node_output?
			layer_outputs.append(np.around(node_output[0], decimals=4))
		if layer != 'output':
			print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
		layer_inputs = layer_outputs
	
	network_predictions = layer_outputs
	return network_predictions
	

if __name__ == "__main__":
	small_network = initialize_network(5, 3, [3,2,3], 1)

	np.random.seed(12)
	inputs = np.around(np.random.uniform(size=5), decimals=2)
	print('The inputs to the network are {}'.format(inputs))

	'''
	node_weights = small_network['layer_1']['node_1']['weights']
	node_bias = small_network['layer_1']['node_1']['bias']

	weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
	node_output = node_activation(weighted_sum)
	'''

	output = forward_propagation(small_network, inputs)
	print('The predicted value by the network for the given input is {}'.format(np.around(output, decimals=4)))
