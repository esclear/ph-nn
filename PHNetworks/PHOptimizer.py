# Optimizer for TensorFlow models utilizing Port-Hamiltonian systems
# Copyright (C) 2020  Daniel Albert
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
    A TensorFlow Optimizer for neural networks based on the paper
    'Port-Hamiltonian Approach to Neural Network Training'
    by Massaroli, Poli, et. al.
    arXiv:1909.02702v1 [cs.NE] – https://arxiv.org/abs/1909.02702v1

    This can't quite be used in-place instead of the optimizers provided in
    TensorFlow, since the interfaces are not compatible.
    However, an effort was made to keep the user interface almost the same.
    Instead of compiling a model with this optimizer and calling model.fit(),
    compile the model (without any optimizer) and pass it to the train() method
    of an instance of this optimizer.

    This is the second iteration, using the velocity verlet method for integration.
"""
# For type anotations
from typing import Callable, List

# For functionality
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks_module
import numpy as np

# custom Verlet integrator
from .verlet_integrator import VerletIntegrator

class PortHamiltonianOptimizer:
    def __init__(
            self,
            alpha: float = 1.0,
            beta: float = 0.0,
            gamma: float = 1.0,
            resistive: float = 0.5,
            ivp_period: float = 1.0,
            ivp_step_size: float = 0.02,
        ):
        """
        Initialize a new Optimizer object

        This creates a new instance of the optimizer and sets the optimizer's
        hyperparameters.

        Parameters
        ----------
        alpha : float
            Parameter α in the Hamiltonian
        beta : float
            Parameter β in the Hamiltonian
        gamma : float
            Parameter γ in the Hamiltonian, represents the mass of every parameter
            and defines the matrix M as M = γ·𝕀
        resistive : float
            Defines the resistive matrix B as B = resistive·𝕀
        ivp_period : float
            Period for which the PH system will be integrated [0, ivp_period]
        ivp_step_size : float
            Step size the integrator shall take
        """

        # Optimizer hyperpameters
        self.alpha     = tf.constant(alpha / 2.0, dtype='float64')
        self.beta      = beta
        self.gamma     = tf.constant(gamma, dtype='float64')
        self.resistive = tf.constant(resistive, dtype='float64')

        # Configuration of the IVP solver
        self.ivp_period    = ivp_period
        self.ivp_step_size = ivp_step_size

        # Initialize internal parameters
        self._last_model = None
        self._param_count = None

    def _check_model_and_state(self, model: keras.models.Model):
        """
            Internal hook to be called on training, this checks whether the model
            being trained was changed

            This (re)creates the matrices B, B⁻¹, M, M⁻¹ and also a function
            returning the inverse of [𝕀 + ½·Δt·M⁻¹·B]⁻¹
            (See verlet_integrator for details)
        """

        if model != self._last_model:
            # Set new model
            self._last_model = model

            param_count = _model_param_count(model)

            self._param_count = param_count

            self.B     =     self.resistive * tf.sparse.eye(param_count, dtype='float64')
            self.B_inv = 1 / self.resistive * tf.sparse.eye(param_count, dtype='float64')

            self.M     =     self.gamma * tf.sparse.eye(param_count, dtype='float64')
            self.M_inv = 1 / self.gamma * tf.sparse.eye(param_count, dtype='float64')

            # This function calculates [𝕀 + ½·Δt·M⁻¹·B]⁻¹
            self.ITMB_inv = lambda Δt: \
                tf.constant(1 / (1 + Δt * self.resistive / (2 * self.gamma))) \
                    * tf.sparse.eye(param_count, dtype='float64')

            # The gradient depends on the current batch, which we don't know at
            # this time. It is set correctly in the train_batch() function.
            stub_loss_gradient = lambda _: tf.zeros((param_count, 1), dtype='float64')

            self._integrator = VerletIntegrator(
                stub_loss_gradient,
                self.M, self.M_inv, self.B, self.B_inv, self.ITMB_inv
            )

    # Training related functionality
    def train(
            self,
            model: keras.models.Model,
            input_data,
            target_data,
            epochs: int = 1,
            batch_size: int = 64,
            shuffle: bool = True,
            callbacks = None,
            metrics: List[tf.metrics.Metric] = [],
        ) -> None:
        """
        Train the model on the given data

        Parameters
        ----------
        model : keras.models.Model
            The tf.keras Model to train
        input_data : tf.Tensor
            Tensor of input samples to train the model on
        target_data : tf.Tensor
            Tensor of expected outputs corresponding to the inputs given in input_data
        epochs : int
            The number of epochs (iterations over the complete training dataset)
            to use for the training of the model given.
        batch_size : int
            Number of items in a batch, 64 by default.
            Use 1 for the '' method of training and the size of the training dataset
            for the '' method described in the paper.
        shuffle : bool
            Whether samples shall be shuffled before batching in different epochs.
            This is supposed to increase accuracy when training over multiple epochs,
            since the batches used for training are (probably) different in every
            epoch, which prevents a reduction of the training's effectiveness
            (compare 'Deep Learning' p. 280)
        callbacks : List of tf.keras.callbacks.Callback objects
            Callbacks to be called during the training process
        metrics : List[tf.metrics.Metric]
            Metrics to calculate during training and report
            These are reported to callbacks and thus also shown in a progress bar
        """

        # Determine the number of samples within the training dataset
        sample_cnt = input_data.shape[0]
        assert target_data.shape[0] == sample_cnt

        train_dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))
        train_dataset = train_dataset.shuffle(
            buffer_size=1024, reshuffle_each_iteration=shuffle
        ).batch(batch_size)

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True, add_progbar=True, model=model,
                verbose=True, epochs=epochs,
                # And, since the tensorflow progbar is a bit broken … with cast
                steps=tf.constant(int(sample_cnt / batch_size), dtype='float64')
            )

        # Begin training
        callbacks.on_train_begin()

        for epoch in tf.range(epochs, dtype='int64'):
            callbacks.on_epoch_begin(epoch)

            for m in metrics:
                m.reset_states()

            for index, batch in train_dataset.enumerate():
                callbacks.on_train_batch_begin(
                    index, {'batch': index, 'size': batch_size}
                )
                loss, energy = self.train_batch(model, *batch, metrics)

                # Call callbacks (also updates progress bar)
                logs = dict(
                    [('loss', loss), ('energy', energy)] +
                    [(m.name, m.result()) for m in metrics]
                )
                # Another cast to work around tf quirks:
                callbacks.on_train_batch_end(tf.cast(index, dtype='float64'), logs)

            callbacks.on_epoch_end(epoch, logs)

        callbacks.on_train_end()

    def train_batch(
            self, model: keras.models.Model,
            input_batch: tf.Tensor, target_batch: tf.Tensor,
            metrics=[]
        ) -> float:
        """
            Train the model with a single batch of samples

            Parameters
            ----------
            model : tf.keras.Model
                The model to train
            input_batch : tf.Tensor
                Input samples of the batch to train.
                These contain the inputs of the batch
            target_batch : tf.Tensor
                Output samples of the batch to train.
                These contain the expected output of the model for the inputs.
            metrics : List[tf.metrics.Metric]
                List of tensorflow metrics.
                These will be evaluated after the training.
        """

        # Basic sanity check that input sample count matches target sample
        # count
        sample_cnt = input_batch.shape[0]
        assert sample_cnt == target_batch.shape[0]

        # Call the model handler
        # (no-op if the model didn't change since the last iteration)
        self._check_model_and_state(model)

        # Redefine hamiltonian and gradient for the current batch
        batch_hamiltonian = self.get_hamiltonian(model, input_batch, target_batch)

        @tf.function
        def loss_gradient(params: tf.Tensor) -> tf.Tensor:
            _model_set_flat_variables(model, params)

            with tf.GradientTape() as tape:
                tape.watch(input_batch)

                loss = model.loss(target_batch, model(input_batch, training=True)) \
                     + self.beta  * tf.tensordot(params, params, 2)

            return _flatten_variables(
                tape.gradient(loss, model.trainable_variables)
            )
        self._integrator.loss_gradient = loss_gradient

        params = _flatten_variables(model.trainable_variables)

        params, velocity = self._integrator.integrate(
            self.ivp_period, self.ivp_step_size, params
        )
        momenta = tf.sparse.sparse_dense_matmul(self.M, velocity)

        _model_set_flat_variables(model, params)

        # Predict output for the current batch and update the metrics accordingly
        prediction = model(input_batch)
        for metric in metrics:
            metric.update_state(target_batch, prediction)

        # We return the loss for the current batch and the energy in the system
        # for the current batch
        return model.loss(target_batch, prediction), batch_hamiltonian(params, momenta)

    def get_hamiltonian(
            self, model: keras.models.Model, inputs: tf.Tensor, targets: tf.Tensor
        ) -> Callable[[tf.Tensor], float]:
        """
            Generate a Tensorflow instrumented function that represents the
            system's Hamiltonian function.
        """
        p = _model_param_count(model)

        return lambda params, momenta: \
            self._hamiltonian(model, inputs, targets, params, momenta)

    @tf.function
    def _hamiltonian(
            self, model: keras.models.Model,
            inputs: tf.Tensor, targets: tf.Tensor,
            params: tf.Tensor, momenta: tf.Tensor
        ) -> float:
        """
            Calculate the value of the Hamiltonian of the system.

            The parameter state should be a tf.Tensor
            (x in the thesis, ξ in the original paper).

            It takes two arguments, params and momenta, instead of the single state
            argument, since it is possible to faster calculate the partial derivative
            with regard to the momenta.
        """

        # Assign state parameters to model
        _model_set_flat_variables(model, params)

        # Predict the outcome with the new parameters
        prediction = model(inputs)

        return 0.5 * (
              self.alpha * model.loss(targets, prediction)
            + self.beta  * tf.tensordot(params, params, 2)
            + self.gamma * tf.tensordot(momenta, momenta, 2)
        )

####################################
# MODEL VARIABLE RELATED FUNCTIONS #
####################################
@tf.function
def _flatten_variables(variables: List[tf.Variable]) -> tf.Tensor:
    """
        Given a list of TensorFlow variable tensors, create a one-dimensional
        tensor from the variables.
    """
    # This is done by concatenating the variables after reshaping them into a single
    # dimension
    return tf.concat([tf.reshape(var, [-1, 1]) for var in variables], 0)


def _model_param_count(model: keras.models.Model) -> int:
    """Count the number of (trainable) parameters of the given model."""
    return sum([tf.size(var) for var in model.trainable_variables])


def _model_set_flat_variables(model: keras.models.Model, values: tf.Tensor) -> None:
    """Set the model's variables to the values provided."""
    variables = model.trainable_variables

    offset = 0
    for variable in variables:
        count = tf.size(variable)
        variable.assign(tf.reshape(values[offset : offset + count], variable.shape))
        offset += count