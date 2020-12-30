import tensorflow as tf
tfkl = tf.keras.layers


class GatedDense(tfkl.Layer):
    """GatedDense layer subclass

    Use a combintation of two Dense layers with one activated and one not.
    Then component-wise multiply the results of both.
    """

    def __init__(self, size, name, activation=None, **kwargs):
        super(GatedDense, self).__init__(name=name, **kwargs)
        self.linear_1 = tfkl.Dense(size, kernel_initializer="he_normal")
        self.linear_2 = tfkl.Dense(size)
        self.activation = activation

    def call(self, inputs):

        x1 = self.linear_1(inputs)
        if self.activation:
            x1 = self.activation(x1)

        x2 = self.linear_2(inputs)
        x2 = tf.nn.sigmoid(x2)

        return tfkl.multiply([x1, x2])
