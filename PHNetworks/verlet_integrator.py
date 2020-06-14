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

import tensorflow as tf

class VerletIntegrator:
    """
        A verlet integrator written for Port-Hamiltonian systems used to train neural networks

        [N.B. the comments below are best read with a text editor that supports proper unicode symbols,
         including diacritics, such as codium or vs code]

        We assume that the Hamiltonian is of the form
            H(x) = L(ϑ) + ½·ωᵀ·M⁻¹·ω.
        Where M = Mᵀ ≻ 0, x = (ϑ,ω) with ω = M·ϑ̇.
        L is the loss function of the neuronal network, with respect to the networks' parameters ϑ.

        This implies that
            ∇ϑ H(x) = ∇ϑ L(ϑ),
            ∇ω H(x) = M⁻¹·ω = ϑ̇.

        The dynamics of the sytem are given by
            ϑ̇ =             ∇ω H(x),
            ω̇ = -∇ϑ H(x) -B·∇ω H(x)
        where B = Bᵀ ≻ 0.

        We can rewrite this as
            ω̇ = M·ϑ̈ = -∇ϑ H(x) -B·∇ω H(x)
        Thus, we can rewrite it as a second order differential equation:
            ϑ̈ = M⁻¹·[- ∇ϑ H(x) - B·∇ω H(x)]
              = M⁻¹·[- ∇ϑ L(ϑ) - B·M⁻¹·ω]
              = M⁻¹·[- ∇ϑ L(ϑ) - B·ϑ̇]                        (S)

        We want to use the velocity verlet method for integration.
        This method is given by the steps
            ϑ(t + Δt) = ϑ(t) + Δt·ϑ̇(t) + ½·(Δt)²·ϑ̈(t)        (A)
            ϑ̇(t + Δt) = ϑ̇(t) + ½·Δt·[ϑ̈(t) + ϑ̈(t + Δt)]       {B}

        However, in {B} ϑ̈(t + Δt) depends on ϑ̇(t + Δt).
        But we can rectify this by solving for ϑ̈(t + Δt):
            ϑ̇(t + Δt) = ϑ̇(t) + ½·Δt·[ϑ̈(t) + ϑ̈(t + Δt)] = ϑ̇(t) + ½·Δt·[ϑ̈(t) + M⁻¹·[- ∇ϑ L(ϑ(t + Δt)) - B·ϑ̇(t + Δt)]]
                      = ϑ̇(t) + ½·Δt·[ϑ̈(t) - M⁻¹·∇ϑ L(ϑ(t + Δt)) - M⁻¹·B·ϑ̇(t + Δt)]
          ⇔ [ϑ̇(t + Δt) + ½·Δt·M⁻¹·B·ϑ̇(t + Δt)] = ϑ̇(t) + ½·Δt·[ϑ̈(t) - M⁻¹·∇ϑ L(ϑ(t + Δt))]
          ⇔ [𝕀 + ½·Δt·M⁻¹·B]·ϑ̇(t + Δt) = ϑ̇(t) + ½·Δt·[ϑ̈(t) - M⁻¹·∇ϑ L(ϑ(t + Δt))]
          ⇔ ϑ̇(t + Δt) = [𝕀 + ½·Δt·M⁻¹·B]⁻¹·[ϑ̇(t) + ½·Δt·[ϑ̈(t) - M⁻¹·∇ϑ L(ϑ(t + Δt))]]     (B)

        Since both M and B are symmetric and positive definite, [𝕀 + ½·Δt·M⁻¹·B]⁻¹ exists.
        We let the user use arbitrary such matrices M and B, however in most cases these will be diagonal matrices,
        such that it is trivial to determine M⁻¹, B⁻¹ and even [𝕀 + ½·Δt·M⁻¹·B]⁻¹.

        This rather minimal implementation defers the calculation of the inverse matrices to the user, since they
        will have the best insights into the properties of M and B which may facilitate the computation of the inverse.
    """
    def __init__(self, loss_gradient, M, M_inv, B, B_inv, ITMB_inv):
        """
            Initialize the verlet integrator

            Parameters
            ----------

            loss_gradient : Function
                Function returning ∇ϑ L(ϑ) given parameters ϑ

            M : tf.tensor
                The "mass" matrix M.
                This must be symmetric and positive definite (M = Mᵀ ≻ 0) by convention.

            M_inv : tf.tensor
                M⁻¹, the inverse of M.

            B : tf.tensor
                The "resistive" or "dissipative" matrix B.
                This must be symmetric and positive definite (B = Bᵀ ≻ 0) by convention.

            B_inv : tf.tensor
                B⁻¹, the inverse of B.

            ITMB_inv : Function
                Function which returns [𝕀 + ½·Δt·M⁻¹·B]⁻¹ when given Δt
        """
        # Store provided parameters
        self.loss_gradient = loss_gradient

        self.M = M
        self.M_inv = M_inv

        self.B = B
        self.B_inv = B_inv

        self.ITMB_inv_fn = ITMB_inv

        # The dimension of the system, i.e. of the position / velocity
        self.dimension = M.shape[0]
        # Basic assertions to catch (some) errors
        assert self.M.shape == [self.dimension, self.dimension], "M has invalid dimensions"
        assert self.B.shape == [self.dimension, self.dimension], "The dimension of B does not match M"
        assert self.M_inv.shape == [self.dimension, self.dimension], "Dimensions imply that M_inv can't be the inverse of M"
        assert self.B_inv.shape == [self.dimension, self.dimension], "Dimensions imply that B_inv can't be the inverse of B"

        # Prepare position and velocity vector, where position corresponds to the network parameters and velocity to the impulse
        self.position         = tf.Variable(tf.zeros((self.dimension, 1), dtype='float64'))
        self.velocity         = tf.Variable(tf.zeros((self.dimension, 1), dtype='float64'))
        self.acceleration     = tf.Variable(tf.zeros((self.dimension, 1), dtype='float64'))

        # This technically does not /have/ to be a member of this class, however doing so we can avoid having compute the gradient
        # twice and to allocate memory and instantiate it in _velocity_verlet_step(), which is the function that will be called
        # most often.
        self.gradient         = tf.Variable(tf.zeros((self.dimension, 1), dtype='float64'))

        # Store the step size
        self.Δt = tf.Variable(-1, dtype='float64')
        # We will store [𝕀 + ½·Δt·M⁻¹·B]⁻¹ here:
        self._itmb_inv = tf.sparse.eye(self.dimension, dtype='float64')


    def _update_stepsize(self, Δt):
        # If the step size hasn't change, we don't need to do anything
        assert Δt > 0, "Δt must be positive"

        if self.Δt == Δt:
            return

        self.Δt.assign(Δt)
        self._itmb_inv = self.ITMB_inv_fn(Δt)


    @tf.function
    def _velocity_verlet_step(self):
        """
            A step of the verlocity verlet method.
            We use the fomulas presented and derived above:

            ϑ(t + Δt) = ϑ(t) + Δt·ϑ̇(t) + ½·(Δt)²·ϑ̈(t)                                   (A)
            ϑ̇(t + Δt) = [𝕀 + ½·Δt·M⁻¹·B]⁻¹·[ϑ̇(t) + ½·Δt·[ϑ̈(t) - M⁻¹·∇ϑ L(ϑ(t + Δt))]]    (B)
            ϑ̈(t + Δt) = M⁻¹·[- ∇ϑ L(ϑ(t + Δt)) - B·ϑ̇(t + Δt)]
        """
        # Determine the new position via (A)
        self.position.assign(self.position + self.Δt * self.velocity + 0.5 * tf.square(self.Δt) * self.acceleration)

        # Determine the new velocity
        self.gradient.assign(self.loss_gradient(self.position))
        self.velocity.assign(
            tf.sparse.sparse_dense_matmul(
                self._itmb_inv,
                (
                    self.velocity
                    + 0.5 * self.Δt * (
                        self.acceleration
                        - tf.sparse.sparse_dense_matmul(self.M_inv, self.gradient)
                    )
                )
            )
        )

        # Calculate the acceleration we would normally have used to calculate the velocity in (S),
        # since we use it as ϑ̈(t) in (B) in the next iteration
        self.acceleration.assign(
            tf.sparse.sparse_dense_matmul(self.M_inv, (- self.gradient - tf.sparse.sparse_dense_matmul(self.B, self.velocity)))
        )


    def integrate(self, t, Δt, initial_position=None, initial_velocity=None):
        """
            Integrate the system over the interval [0, t], taking steps of size Δt.

            For simplicity, the implementation is such that if t is not a multiple of Δt, this implementation will
            actually integrate over the interval [0, k * Δt], such that k is minimal with k * Δt ≥ t.

            Parameters
            ----------
            t : float
                The duration over which the differential equation shall be integrated.
            Δt : float
                Step size to use for the integration
            initial_position : tf.tensor
                The initial position (parameters) of the neural network.
                Uses the last known parameters if not specified.
            initial_velocity : tf.tensor
                The initial velocity (given by the impulse) of the state.
                Uses the last known velocity if not specified.
        """
        self._update_stepsize(Δt)

        if initial_position is not None:
            self.position.assign(initial_position)
        if initial_velocity is not None:
            self.velocity.assign(initial_velocity)

        s = tf.Variable(0, dtype='float64')
        while s < t:
            self._velocity_verlet_step()
            s.assign(s + Δt)

        return self.position, self.velocity