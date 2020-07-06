# Port-Hamiltonian Neural Networks
This is a TensorFlow implementation of Port-Hamiltonian Neural Networks and their training based on the methods described in the paper [Port-Hamiltonian Approach to Neural Network Training](https://arxiv.org/abs/1909.02702v1) by Stefano Massaroli, Michael Poli, et.al. (arXiv:1909.02702v1 [cs.NE]).

It is written for TensorFlow 2.1.0, but it may work with future versions as well.
A main focus in development was usability and flexibility.
The goal was that this optimizer works with all regular models, i.e. models to which other, more popular methods, such as SGD, could be applied as well.

## Ergonomics
Due to API limitations of the TensorFlow library, it is unfortunately not possible to pass the custom optimizer to Keras' `model.compile()` function and use `model.fit()` to train the model afterwards.
The problem here is that with the Keras optimizer interface, the optimizer only ever gets to see the gradient of the current batch for the current set of model parameters.
However, it does not have access to the batch itself, which we need for the hamiltonian.
It is also not possible to use the hamiltonian as loss function, since the gradient needs to be evaluated at multiple points (i.e. for multiple parameters) by the IVP solvers.

# Integration
This algorithm uses a custom written and optimized velocity verlet implementation completely written as tensorflow code.
Its implementation and derivation is described in [`PHNetworks/verlet_integrator.py`](PHNetworks/verlet_integrator.py).

# Dependencies
This optimizer has been developed using TensorFlow 2 nightly and some of the examples use TensorBoard 2 nightly using Python 3.8.  
The latest version under which the optimizer was tested and working can be found in the [requirements.txt](requirements.txt) file and may be installed using `pip -r requirements.txt`.

The latest TensorFlow 2 nightly version known to work with this implementation is `tf-nightly==2.4.0.dev20200703`.

## Example
The following example shows how a model can be trained for the ubiquitous MNIST dataset using this optimizer:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PHNetworks.PHOptimizer import PortHamiltonianOptimizer as PHOpt

def make_model():
    keras.backend.set_floatx('float64')
    model = keras.models.Sequential(
        layers=[
            # Make 28x28-entry two-dimensional input 784-entry one-dimensional
            layers.Flatten(input_shape=(28, 28), name='input_layer'),
            # Hidden layer: 32 nodes with ReLU activation function
            layers.Dense(32, activation='relu', name='hidden_layer'),
            layers.Dropout(0.2),
            # Output layer: 10 nodes for 10 possible digits
            layers.Dense(10, activation='softmax', name='output_layer')
        ],
        name='mnist_model')
    model.summary()
    return model

model = make_model()
optimizer = PHOpt()
model.compile(loss=keras.losses.CategoricalCrossentropy())

# Load MNIST handwritten digits dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Prescale pixel byte brightness to float in range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert integer category to activation of the output neurons
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

# Train the model
optimizer.train(model, x_train, y_train, epochs=4)
```

## Copyright
&copy; 2020 Daniel Albert

This implementation has been written as a part of my [bachelor thesis](https://proj.esclear.de/bachelor-thesis).  
As such I provided it under the terms of the [GNU General Public License v3.0](LICENSE).
