# Tensorflow callback for plotting with matplotlib
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
from tensorflow.keras.callbacks import Callback

import numpy as np
import matplotlib.pyplot as mp

from PHNetworks.PHOptimizer import _flatten_variables

class Plotter(Callback):
    def __init__(self, step_size, model=None, integrator=None):
        super(Plotter, self).__init__()
        
        self.step_size   = step_size
        
        self.plot_model    = model is not None
        self.model         = model
        self.plot_velocity = integrator is not None
        self.integrator    = integrator
        
        self.time_steps  = []
        self.hamiltonian = []
        self.loss_values = []
        self.accuracy    = []
        
        self.parameters  = []
        self.velocities  = []
    
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        self.time_steps.append(float(epoch) * self.step_size)
        
        self.hamiltonian.append(logs['energy'])
        self.loss_values.append(logs['loss'])
        
        self.accuracy.append(logs['categorical_accuracy'])
        
        if self.plot_model:
            self.parameters.append(
                _flatten_variables(self.model.trainable_variables).numpy().reshape((-1, 1))
            )
        if self.plot_velocity:
            self.velocities.append(self.integrator.velocity.numpy())
        
    def on_train_end(self, logs=None):
        self.time_steps.append(self.time_steps[-1] + self.step_size)
        self.hamiltonian.append(self.hamiltonian[-1])
        self.loss_values.append(self.loss_values[-1])
        self.accuracy.append(self.accuracy[-1])
        
        if self.plot_model:
            self.parameters.append(self.parameters[-1])
        
        if self.plot_velocity:
            self.velocities.append(self.velocities[-1])
        
        self.plot()
    
    def plot(self):
        t = np.array(self.time_steps)
        
        H = np.array(self.hamiltonian)
        L = np.array(self.loss_values)
        A = np.array(self.accuracy)

        fig = mp.figure(figsize=(11,5))

        ax_hamiltonian = mp.subplot(3, 1, 1, autoscale_on=True)
        ax_hamiltonian.set_yscale("log")
        ax_hamiltonian.margins(x=0)
        H_plot = mp.plot(t, H, '-', c='#00aecf', label=r'$\mathcal{H}$')
        E_plot = mp.plot(t, L, '-', c='#489324', label=r'$E(S; \vartheta)$')
        mp.ylabel(r'$\mathcal{H}, E(S; \vartheta)$')
        
        ax_accuracy = ax_hamiltonian.twinx()
        
        A_plot = mp.plot(t, A, '-', c='#eb690b', label=r'Genauigkeit')
        mp.ylabel(r'Genauigkeit')
                
        plots  = H_plot + E_plot + A_plot
        labels = [p.get_label() for p in plots]
        mp.legend(plots, labels, loc='center right')
        mp.title('Trainingsverlauf')
        
        if self.plot_model:
            theta = np.concatenate(self.parameters, axis=1).T
            ax_params = mp.subplot(3, 1, 2, sharex=ax_hamiltonian)
            ax_params.set_yscale("symlog")
            mp.plot(t, theta, '-')
            mp.ylabel(r'$\vartheta$')
        
        if self.plot_velocity:
            velos = np.concatenate(self.velocities, axis=1).T
            ax_velos = mp.subplot(3, 1, 3, sharex=ax_hamiltonian)
            ax_velos.set_yscale("symlog")
            mp.plot(t, velos, '-')
            mp.ylabel(r'$\dot\vartheta$')

        mp.xlabel('Zeit $t$')
        
        fig.tight_layout()
        
        return fig