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

"""A TensorFlow Optimizer for neural networks based on the paper
    'Port-Hamiltonian Approach to Neural Network Training'
    by Massaroli, Poli, et. al.
    arXiv:1909.02702v1 [cs.NE] – https://arxiv.org/abs/1909.02702v1

    This can't be used in-place instead of the optimizers provided in
    TensorFlow, since the interfaces are not compatible.
"""
# For type anotations
from typing import List

# For functionality
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks_module
from scipy import sparse
from scipy.integrate import solve_ivp
import numpy as np


class PortHamiltonianOptimizer:
    def __init__(
            self,
            alpha: float = 1.0,
            beta: float = 0.0,
            gamma: float = 1.0,
            resistive: float = 0.5,
            ivp_period: float = 1.0,
            ivp_steps: int = None,
            ivp_solver: str = 'RK45',
        ):
        """TODO: DOCUMENTATION"""

        # Optimizer hyperpameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.resistive = resistive

        # Configuration of the IVP solver
        self.ivp_period = ivp_period
        self.ivp_solver = ivp_solver
        self.ivp_t_span = np.linspace(0.0, self.ivp_period, ivp_steps) if ivp_steps else (0.0, self.ivp_period)

        # Initialize internal parameters
        self._last_model = None
        self._momenta = None
        self._param_count = None
        self._dynamics_matrix = None

        # TODO: TEMPORARY
        self.hamiltonian = None

    def _check_model_and_state(self, model: keras.models.Model):
        """Hook to be called on training, this checks whether the model being trained was changed"""

        if model != self._last_model:
            # Set new model
            self._last_model = model

            param_count = _model_param_count(model)

            self._param_count = param_count
            self._momenta = tf.zeros([param_count], dtype='float64')

            B = self.resistive * sparse.eye(param_count)
            self._dynamics_matrix = PortHamiltonianOptimizer._calculate_dynamics_matrix(B)
            del B

    # Training related functionality
    def train(
            self,
            model: keras.models.Model,
            input_data,
            target_data,
            epochs: int = 1,
            batch_size: int = 64,
            shuffle: bool = True,
            callbacks=None,
            metrics=[],
            hamiltonian_generator=None
        ) -> None:
        """TODO: DOCUMENTATION"""

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
                verbose=True, epochs=epochs, steps=sample_cnt)

        # Begin training
        callbacks.on_train_begin()

        for epoch in tf.range(epochs):
            callbacks.on_epoch_begin(epoch)

            for m in metrics:
                m.reset_states()

            for index, batch in train_dataset.enumerate():
                callbacks.on_train_batch_begin(index * batch_size, {'batch': index, 'size': batch_size})
                loss, energy = self.train_batch(model, *batch, metrics, hamiltonian_generator)

                # Call callbacks (also updates progress bar)
                callbacks.on_train_batch_end(tf.cast(index * batch_size, tf.float64), dict([('loss', loss), ('energy', energy)] + [(m.name, m.result()) for m in metrics]))

            callbacks.on_epoch_end(tf.cast(epoch, tf.int64), dict([('loss', loss), ('energy', energy)] + [(m.name, m.result()) for m in metrics]))

        callbacks.on_train_end()

    def train_batch(
            self, model: keras.models.Model, input_batch, target_batch,
            metrics=[], hamiltonian_generator=None
        ) -> float:
        """Train the model with a single batch of samples
        """

        # Basic sanity check that input sample count matches target sample
        # count
        sample_cnt = input_batch.shape[0]
        assert sample_cnt == target_batch.shape[0]

        self._check_model_and_state(model)
        p = self._param_count

        params = _flatten_variables(model.trainable_variables)
        state = tf.concat([params, self._momenta], 0)

        dynamics, hamiltonian = self.get_dynamics(model, input_batch, target_batch, hamiltonian_generator)

        # Solve the IVP problem using SciPy's IVP solver.
        solution = solve_ivp(dynamics, self.ivp_t_span, state, method=self.ivp_solver)
        # TODO: Check solution

        # The new state resulting from the evolution of the IVP
        new_state = solution.y[:, -1]

        # Split the network parameters and evolution momenta from the new state
        params = new_state[:p]
        momenta = new_state[p:]

        _model_set_flat_variables(model, params)
        self._momenta = momenta

        # Predict output for the current batch and update the metrics accordingly
        prediction = model(input_batch)
        for metric in metrics:
            metric.update_state(target_batch, prediction)

        # We return the loss for the current batch and the energy in the system for the current batch
        return model.loss(target_batch, prediction), hamiltonian(params, momenta)

    def get_dynamics(
            self, model: keras.models.Model, inputs, targets, hamiltonian_generator=None
        ):
        """This function returns a tf.function that defines the dynamics of a PH system.
        These are defined as ẋ=(J-R)·∇H(x)=F·∇H(x), where F the dynamics matrix calculated when a new model is loaded."""

        self._check_model_and_state(model)

        p = self._param_count
        F = self._dynamics_matrix

        if not self.hamiltonian:
            self.hamiltonian = (hamiltonian_generator or self.get_hamiltonian)(model, inputs, targets)
        hamiltonian = self.hamiltonian

        # As long as no custom hamiltonian function is given, the first
        # dynamics function is used, which has some features of the
        # gradient calculated by hand, which results in a speed
        # improvement of about 390 % in tests.
        # In case a custom hamiltonan function is used, the gradient
        # must be determined by the gradient tape as well, which is
        # slower.
        if hamiltonian_generator is None:
            # This is the function which describes the parameter dynamics and
            # will be used for integration
            def dynamics(t, state):
                state = tf.convert_to_tensor(state)

                params = state[:p]
                momenta = state[p:]

                _model_set_flat_variables(model, params)

                with tf.GradientTape() as tape:
                    tape.watch(state)

                    loss = model.loss(targets, model(inputs))

                loss_gradient = _flatten_variables(
                    tape.gradient(loss, model.trainable_variables)
                )

                # The hamiltonian is H(x) = H(ϑ,ω) = ½·( α·loss(ϑ) + β·∥ϑ∥² + γ·∥ω∥² ) = ½·( α·loss(ϑ) + β·ϑᵀ·ϑ + γ·ωᵀ·ω )
                #   Thus, ∇_ϑ H = ½·α·∇_ϑ loss(ϑ) + β·ϑ
                #   Also, ∇_ω H = γ·ω
                # We thus obtain the following derivatives:
                nabla_theta = 0.5 * self.alpha * loss_gradient + self.beta * params
                nabla_omega = self.gamma * momenta

                # The gradient is just the concatenation ∇H = (∇_ϑ H, ∇_ω H)
                gradient = tf.concat([nabla_theta, nabla_omega], 0)

                out = F * gradient.numpy()

                return out

        else:
            # This is the function which describes the parameter dynamics and
            # will be used for integration
            def dynamics(t, state):
                state = tf.convert_to_tensor(state)

                params = state[:p]
                momenta = state[p:]

                _model_set_flat_variables(model, params)

                # Determine the gradient of the hamiltonian
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(momenta)

                    system_energy = hamiltonian(params, momenta)

                grad_theta = tape.gradient(system_energy, model.trainable_variables)
                grad_omega = tape.gradient(system_energy, momenta)

                del tape

                grad_theta = _flatten_variables(grad_theta)
                gradient = tf.concat([grad_theta, grad_omega], 0)

                out = F * gradient.numpy()

                return out

        return dynamics, hamiltonian

    def get_hamiltonian(
            self, model: keras.models.Model, inputs: tf.Tensor, targets: tf.Tensor
        ):
        """Generate a Tensorflow instrumented function that represents the systems Hamiltonian.

        """
        p = _model_param_count(model)

        @tf.function
        def hamiltonian(params: tf.Tensor, momenta: tf.Tensor) -> float:
            """Calculate the value of the Hamiltonian of the system.

            The parameter state should be a tf.Tensor (x in the thesis, Xi in the original paper).

            It takes two arguments, params and momenta, instead of the single state argument,
            since it is possible to faster calculate the partial derivative with regard to the momenta.
            """

            # Assign state parameters to model
            _model_set_flat_variables(model, params)

            # Predict the outcome with the new parameters
            prediction = model(inputs)

            return 0.5 * (
                  self.alpha * model.loss(targets, prediction)
                + self.beta  * tf.tensordot(params, params, 1)
                + self.gamma * tf.tensordot(momenta, momenta, 1)
            )

        return hamiltonian

    @staticmethod
    def _calculate_dynamics_matrix(B: np.ndarray) -> np.ndarray:
        """This function calculates the matrix called F = J - R in the
        paper, where J = [[0, 1],[-1, 0]] and R = [[0, 0],[0, B]]
        (0 and 1 are the zero and identity matrix respectively).
        It is calculated and stored as a sparse tensor in order to save
        memory.

        The parameter B must be a symmetric and positive definite p x p
        matrix. Preferably, B is also a sparse matrix.

        The types are annotated with np.ndarray, although they are
        sparse scipy matrices. However, the matrix and its operations
        are compatible to numpy's operations, thus the annotation.
        """

        # Get one side of B and check that B is actually square
        # We don't check that B is symmetric and positive definite.
        p = B.shape[0]
        assert B.shape == (p, p)

        # Create a p x p identity matrix
        I = sparse.eye(p)
        # Build F from its four components
        F = sparse.bmat([[None, I], [-I, -B]])
        # Remove the identity matrix from memory, since we don't need it
        # anymore.
        del I

        return F


####################################
# MODEL VARIABLE RELATED FUNCTIONS #
####################################
def _flatten_variables(variables: List[tf.Variable]) -> tf.Tensor:
    """Given a list of TensorFlow variable tensors, create a one-dimensional tensor from the variables."""
    # This is done by concatenating the variables after reshaping them into a single dimension
    return tf.concat([tf.reshape(var, [-1]) for var in variables], 0)


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