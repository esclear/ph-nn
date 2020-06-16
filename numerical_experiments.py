#!/usr/bin/env python3

import itertools
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.layer_utils import count_params

from PHNetworks.PHOptimizer import PortHamiltonianOptimizer as PHOpt


DEPTHS = [0, 1, 2, 4, 6]
LAYER_SIZES = [1000, 500, 250, 50]
EPOCHS = 6
IVP_PERIOD = 30.0
IVP_STEP_SIZE = 0.25
BATCH_SIZE = 60000
START_WITH_LARGEST = False

#########################################
# MODEL-CREATION-SPECIFIC FUNCTIONALITY #
#########################################
def get_layer_sizes(depths):
    all_descending_layer_sizes = itertools.chain(*[
            itertools.combinations_with_replacement(LAYER_SIZES, r) for r in depths
    ])
    return all_descending_layer_sizes


def make_model(layer_sizes):
    keras.backend.set_floatx('float64')
    model = keras.models.Sequential(name='mnist_model')
    
    # Make 28x28-entry two-dimensional input 784-entry one-dimensional
    model.add(layers.Flatten(input_shape=(28, 28), name='input_layer'))
    
    # Add hidden layers with respective layer sizes
    for i, size in enumerate(layer_sizes, start=1):
        model.add(layers.Dense(size, activation='sigmoid', name=f'hidden_layer_{i}'))
    
    # Output layer: 10 nodes for 10 possible digits
    model.add(layers.Dense(10, activation='sigmoid', name='output_layer'))

    # Compile the model with additional info
    model.compile(loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


def has_repetitions(iter, count=3):
    last = None
    reps = 0
    for item in iter:
        if item == last:
            reps += 1
            if reps >= count:
                return True
        else:
            last = item
            reps = 1
    return False


#################################
# MODEL TRAINING AND EVALUATION #
#################################
# Load MNIST handwritten digits dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Prescale pixel byte brightness to float in range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert integer category to activation of the output neurons
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)


def train_and_evaluate(sizes, id=None):    
    model = make_model(sizes)

    label = f'{id:03d}_784-{"".join([f"{size}-" for size in sizes])}10_{count_params(model.trainable_weights)}'
    logdir = f'arch_eval_logs/{label}'
    
    optimizer = PHOpt(ivp_period=IVP_PERIOD/EPOCHS, ivp_step_size=IVP_STEP_SIZE)
    
    print(f'DESC: Training model {label}')
    start = timer()
    
    with tf.device('/gpu:0'):
        optimizer.train(
            model, x_train, y_train,
            batch_size=BATCH_SIZE,
            metrics=[keras.metrics.CategoricalAccuracy()],
            callbacks=[
            #    tf.keras.callbacks.TensorBoard(logdir, update_freq=4, profile_batch=0)
            ],
            epochs=EPOCHS
        )
    
    end = timer()
    print(f'TIME: Training took {end - start:.2f}s')
    
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'ACCU: Reached accuracy of {accuracy * 100:02.2f}%')

    print(f'>>> {id:d},{len(sizes)+1:d},784-{"".join([f"{size}-" for size in sizes])}10,{count_params(model.trainable_weights)},{accuracy * 100:02.2f}%,{end - start:.2f}s')
    
    del model


def exec():
    layer_sizes = get_layer_sizes(DEPTHS)
    layer_sizes = itertools.filterfalse(has_repetitions, layer_sizes)
    layer_sizes = list(layer_sizes)
    layer_sizes = sorted(layer_sizes, key=sum)
    layer_sizes = list(enumerate(layer_sizes))
    if START_WITH_LARGEST:
        layer_sizes = list(reversed(layer_sizes))
    
    print(f'COUN: Training {len(layer_sizes)} networks')
    print('>>> run,layer count,layers,param count,accuracy,time')
    
    for id, layer in layer_sizes:
        train_and_evaluate(layer, id)


if __name__ == '__main__':
    exec()
