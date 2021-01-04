from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers


class PseudoInputs(tfkl.Layer):
    def __init__(self, n_pseudo_inputs=500):
        super().__init__()
        self.n_pseudo_inputs = n_pseudo_inputs
        self.pseudo_inputs = None

    def call(self, inputs=None):
        # abstract method
        pass

    def get_n(self):
        return self.n_pseudo_inputs


class PInputsGenerated(PseudoInputs):
    def __init__(self, pseudo_inputs_mean=0.0,
                 pseudo_inputs_std=0.01, original_dim=(28, 28), n_pseudo_inputs=500):
        super().__init__(n_pseudo_inputs)
        self.pseudo_input_std = pseudo_inputs_std
        self.pseudo_input_mean = pseudo_inputs_mean
        self.pre_pseudo_inputs = tf.eye(n_pseudo_inputs)
        self.pseudo_inputs_layer = tfkl.Dense(
            np.prod(original_dim),
            kernel_initializer=tfk.initializers.RandomNormal(
                mean=pseudo_inputs_mean, stddev=pseudo_inputs_std),
            activation="relu",
            name="pseudo_input_layer"
        )

    def call(self, inputs=None):
        # If the pre pseudo inputs are generated we have to create the new
        # pseudo inputs
        #
        # In case of the "generate" vampprior an additional fully
        # connected layer maps a scalar to an image
        # (e.g in case of mnist [] -> [28, 28, 1] )
        # recompute pseudo inputs if we choose the generate strategy
        return self.pseudo_inputs_layer(self.pre_pseudo_inputs)


class PInputsData(PseudoInputs):
    def __init__(self, pseudo_inputs):
        super().__init__(n_pseudo_inputs=pseudo_inputs.shape[0])
        self.pseudo_inputs = pseudo_inputs

    def call(self, inputs=None):
        # For the "data" vampprior the pseudo inputs are fixed over all epochs
        # and batches.
        return self.pseudo_inputs
