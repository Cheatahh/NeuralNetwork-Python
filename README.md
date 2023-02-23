# NeuralNetwork-Python

Simple neural network implementation written in Python.

See `training.py` for an example using the [mnist](http://yann.lecun.com/exdb/mnist) dataset.

### Example (xor function):

```python
import numpy
from nn import NeuralNetwork

network = NeuralNetwork(2, 5, 1)

dataset = [
    (numpy.array([0, 0]), numpy.array([0])),
    (numpy.array([1, 0]), numpy.array([1])),
    (numpy.array([0, 1]), numpy.array([1])),
    (numpy.array([1, 1]), numpy.array([0]))
]

network.train(dataset, learning_rate=1, epochs=10000)

for inputs, expected in dataset:
    print(expected, network.feed_forward(inputs))
```