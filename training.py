"""
    training example with the mnist dataset
"""

import csv
import sys
import zipfile

import numpy
import pickle

# noinspection PyUnresolvedReferences
from nn import NeuralNetwork, Layer, sigmoid, sigmoid_derivative

ansi_reset = "\033[0m"
ansi_light_blue = "\033[94m"
ansi_light_green = "\033[92m"


def read_csv(file, total):
    dataset = []
    with open(file) as f:
        reader = csv.reader(f)
        element = 0
        for row in reader:
            element += 1
            if element % max(total // 100, 1) == 0:
                sys.stdout.write(f"""\r{ansi_light_blue}reading file{ansi_reset} {file} {
                numpy.around(element / total * 100, 2)}%""")
            result = numpy.zeros(10)
            # first element is the digit label
            result[int(row[0])] = 1
            # csv file contains pixel values from 0-255, normalize them into 0-1
            dataset.append((numpy.array([float(x) / 255.0 for x in row[1:]]), result))
    print()
    return dataset


# unpack the mnist dataset
with zipfile.ZipFile("mnist_dataset.zip", 'r') as zip_reader:
    zip_reader.extractall("./")


# pre-trained with 20 epochs
pickle_file = "pretrained-network.pkl"
try:
    with open(pickle_file, "rb") as pkl_f:
        nn = pickle.load(pkl_f)
except FileNotFoundError:
    # 28 * 28 = 784 inputs
    # 2 hidden layers with 16 neurons each
    # 10 outputs (digits 0-9)
    nn = NeuralNetwork(28 * 28, 16, 16, 10)

"""
mnist_train = read_csv("mnist_train.csv", 60000)
# 20 epochs seems to be the sweet spot for this dataset
nn.train(mnist_train, learning_rate=1, epochs=20, debug=True)
print()
with open(pickle_file, "wb") as pkl_f:
    pickle.dump(nn, pkl_f)
"""

mnist_test = read_csv("mnist_test.csv", 10000)
print(f"{ansi_light_green}accuracy{ansi_reset}: {nn.accuracy(mnist_test) * 100}%\n")

my_own_image = read_csv("my_own_image.csv", 1)
print(f"{ansi_light_green}passes{ansi_reset}: {nn.accuracy(my_own_image) == 1}")
