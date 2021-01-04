import tensorflow as tf


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def on_train_begin(self, logs):
        self.re = []

    def on_epoch_end(self, epoch, logs):
        reconstruction_error = self.model.neg_log_likelihood(
            self.data, self.model.decoder(self.model.encoder((self.data))))
        # self.model.get_kl())
        self.re.append(tf.math.reduce_mean(reconstruction_error))
