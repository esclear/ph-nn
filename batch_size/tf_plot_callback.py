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
    def __init__(self, step_size):
        super(Plotter, self).__init__()
        
        self.step_size     = step_size
        self.current_epoch = 0
        self.total_batches = 0
        
        self.time_steps  = []
        self.hamiltonian = []
        self.loss_values = []
        self.accuracy    = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
    
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.total_batches += 1
        
        self.time_steps.append(float(self.total_batches) * self.step_size)
        
        self.hamiltonian.append(logs['energy'])
        self.loss_values.append(logs['loss'])
        
        self.accuracy.append(logs['categorical_accuracy'])
        
    def on_train_end(self, logs=None):
        self.time_steps.append(self.time_steps[-1] + self.step_size)
        self.hamiltonian.append(self.hamiltonian[-1])
        self.loss_values.append(self.loss_values[-1])
        self.accuracy.append(self.accuracy[-1])
        
        self.plot()
    
    def plot(self):
        t = np.array(self.time_steps)
        
        H = np.array(self.hamiltonian)
        L = np.array(self.loss_values)
        A = np.array(self.accuracy)

        fig = mp.figure(figsize=(11,2.2))

        ax_hamiltonian = mp.subplot(1, 1, 1, autoscale_on=True)
        ax_hamiltonian.set_yscale("log")
        ax_hamiltonian.margins(x=0)
        H_plot = mp.plot(t, H, '-', c='#00aecf', label=r'$\mathcal{H}$')
        E_plot = mp.plot(t, L, '-', c='#489324', label=r'$E(S; \vartheta)$')
        mp.ylabel(r'$\mathcal{H}, E(S; \vartheta)$')

        mp.xlabel('Zeit $t$')
        
        ax_accuracy = ax_hamiltonian.twinx()
        
        A_plot = mp.plot(t, A, '-', c='#eb690b', label=r'Genauigkeit')
        mp.ylabel('Genauigkeit')

        mp.xlabel('Zeit $t$')
                
        plots  = H_plot + E_plot + A_plot
        labels = [p.get_label() for p in plots]
        mp.legend(plots, labels, loc='center right')
        mp.title('Trainingsverlauf')
        
        fig.tight_layout()
        
        return fig