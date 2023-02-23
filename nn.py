"""
    neural network implementation
"""

import sys

import numpy


ansi_reset = "\033[0m"
ansi_white = "\033[97m"


#
# activation functions
#

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return numpy.tanh(x)


def tanh_derivative(x):
    return 1.0 - x**2


def relu(x):
    return numpy.maximum(0, x)


def relu_derivative(x):
    return 1. * (x > 0)


def silu(x):
    return x * sigmoid(x)


def silu_derivative(x):
    return sigmoid(x) + x * sigmoid_derivative(x)


def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()


def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))


#
# neural network implementation
#

class Layer:

    def __init__(self, input_size, neuron_count, activation_function=sigmoid,
                 activation_function_derivative=sigmoid_derivative, weights=None, bias=1):
        self.weights = numpy.random.random((input_size, neuron_count)) * 2 - 1 if weights is None else weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.bias = bias
        self.results = None
        self.delta = None

    def feed_forward(self, inputs):
        self.results = self.activation_function(inputs @ self.weights + self.bias)
        return self.results


class NeuralNetwork:

    def __init__(self, *layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def backpropagation(self, inputs, expected_output, learning_rate):
        output = self.feed_forward(inputs)

        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            if idx == len(self.layers) - 1:
                layer_error = expected_output - output
                layer.delta = layer_error * layer.activation_function_derivative(output)
            else:
                next_layer = self.layers[idx + 1]
                layer_error = next_layer.weights @ next_layer.delta
                layer.delta = layer_error * layer.activation_function_derivative(layer.results)

        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            previous_layer_result = numpy.atleast_2d(inputs if idx == 0 else self.layers[idx - 1].results)
            layer.weights += layer.delta * previous_layer_result.T * learning_rate

    def train(self, dataset, learning_rate, epochs, debug=False):
        for i in range(epochs):
            element = 0
            for inputs, expected in dataset:
                element += 1
                if debug and element % max(len(dataset) // 100, 1) == 0:
                    sys.stdout.write(f"""\r{ansi_white}training epoch{ansi_reset}: {i+1}/{
                    epochs} @ {numpy.around(element / len(dataset) * 100, 2)}%""")
                self.backpropagation(inputs, expected, learning_rate)
        if debug:
            print()

    def accuracy(self, dataset):
        correct = 0
        element = 0
        for inputs, expected in dataset:
            element += 1
            if element % max(len(dataset) // 100, 1) == 0:
                sys.stdout.write(f"""\r{ansi_white}calculating accuracy{ansi_reset} {
                numpy.around(element / len(dataset) * 100, 2)}%""")
            output = self.feed_forward(inputs)
            if classify_outputs(output) == classify_outputs(expected):
                correct += 1
        print()
        return correct / len(dataset)


def classify_outputs(outputs):
    return numpy.argmax(outputs)
