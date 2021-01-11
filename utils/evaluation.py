import tensorflow as tf
import pandas as pd
import os


class HistorySaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, destination):
        super().__init__()
        self.destination = destination

    def on_epoch_begin(self, epoch, logs):
        print(epoch)
        self.save()

    def on_train_end(self, logs):
        self.save()

    def save(self):
        last_values = {}
        for key in self.model.history.history.keys():
            last_values[key] = [self.model.history.history[key][-1]]

        if len(last_values.keys()) == 0: # first epoch, no results yet 
            return

        print("Saving history to", self.destination)

        df_history = pd.DataFrame.from_dict(last_values)

        df_history.to_csv(self.destination, mode='a', header = False, index = False)
