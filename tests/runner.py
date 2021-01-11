import sys
sys.path.append('../')
from enum import Enum
import models.hvae as hvae
import tensorflow as tf
import models.vae as vae
import models.vanilla_vae as vanilla_vae
from utils.datasets import DatasetKey, get_dataset
from utils.pseudo_inputs import PInputsData, PInputsGenerated
import pandas as pd
import os 
import numpy as np

Dataset = DatasetKey


class Architecture(Enum):
    VANILLA = 0
    HVAE = 1


class PriorConfiguration(Enum):
    SG = 0
    VAMPDATA = 1
    VAMPGEN = 2


dataset_key_dict = {
    DatasetKey.MNIST: "MNIST",
    DatasetKey.OMNIGLOT: "OMNIGLOT"
}
architecture_key_dict = {
    Architecture.VANILLA: "VANILLA",
    Architecture.HVAE: "HVAE"
}
prior_key_dict = {
    PriorConfiguration.SG: "SG",
    PriorConfiguration.VAMPDATA: "VAMPDATA",
    PriorConfiguration.VAMPGEN: "VAMPGEN"
}


def get_checkpoint_path(dataset_key, architecture, prior_configuration):
    name = "{root_dir}/{dataset}_{model}_{prior}.cpkt".format(
        root_dir="../checkpoints",
        dataset=dataset_key_dict[dataset_key],
        model=architecture_key_dict[architecture],
        prior=prior_key_dict[prior_configuration]
    )
    return name


def get_history_path(dataset_key, architecture, prior_configuration):
    name = "{root_dir}/{dataset}_{model}_{prior}.csv".format(
        root_dir="../history",
        dataset=dataset_key_dict[dataset_key],
        model=architecture_key_dict[architecture],
        prior=prior_key_dict[prior_configuration]
    )
    return name


class Runner():
    n_pseudo_inputs = 500
    learning_rate = 0.001
    pseudo_inputs = None

    def __init__(
        self,
        dataset_key: DatasetKey,
        architecture: Architecture,
        prior_configuration: PriorConfiguration,
        nb_epochs=2000
    ):
        self.dataset_key = dataset_key
        self.architecture = architecture
        self.prior_configuration = prior_configuration
        self.nb_epochs = nb_epochs
        self.checkpoint_path = get_checkpoint_path(
            dataset_key=dataset_key,
            architecture=architecture,
            prior_configuration=prior_configuration
        )
        self.history_path = get_history_path(
            dataset_key=self.dataset_key,
            architecture=self.architecture,
            prior_configuration=self.prior_configuration
        )

    def fetch_dataset(self):
        (self.x_train, self.x_test) = get_dataset(DatasetKey.MNIST)

    def prepare_model(self):
        if self.dataset_key == DatasetKey.OMNIGLOT:
            self.n_pseudo_inputs = 1000
            self.learning_rate = 0.0005
        if self.prior_configuration != PriorConfiguration.SG:
            self.prior_type = vae.Prior.VAMPPRIOR
            if self.prior_configuration == PriorConfiguration.VAMPDATA:
                self.pseudo_inputs = PInputsData(
                    pseudo_inputs=self.x_train[:self.nb_pseudo_inputs])
            else:
                self.pseudo_inputs = PInputsGenerated(
                    original_dim=self.x_train.shape[1:], n_pseudo_inputs=self.n_pseudo_inputs)
        else:
            self.prior_type = vae.Prior.STANDARD_GAUSSIAN

        self.model_class = vanilla_vae.VanillaVAE if self.architecture == Architecture.VANILLA else hvae.HVAE
        self.model = self.model_class(
            original_dim=self.x_train.shape[1:], prior_type=self.prior_type, pseudo_inputs=self.pseudo_inputs)
        self.model.prepare(learning_rate=self.learning_rate)

    def reload_if_possible(self):
        if not os.path.isfile(self.history_path):
            return # no history stored for this model
        history_losses = pd.read_csv(self.history_path, names = ["epoch", "loss", "val_loss"])
        number_epochs_done = history_losses.shape[0]
        if number_epochs_done != 0:
            # relink all layers
            one_input_size = [1] + list(self.x_train.shape[1:])
            self.model(np.zeros(one_input_size))

            self.model.load_weights(self.checkpoint_path)
            self.nb_epochs -= number_epochs_done

    def run(self):
        self.fetch_dataset()
        self.prepare_model()
        self.reload_if_possible()
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                       min_delta=0.0001,
                                                       patience=50,
                                                       verbose=1,
                                                       restore_best_weights=True)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=False,
                                                         monitor='val_loss',
                                                         verbose=1)
        csv_logger = tf.keras.callbacks.CSVLogger(self.history_path, append=True)

        if self.nb_epochs <= 0:
            print("This model has already been trained and stored for the required number of epochs.")
            print("Delete the file under {history} if you want to retrain.".format(history = self.history_path))
        self.model.fit(self.x_train, self.x_train, epochs=self.nb_epochs,
                       validation_split=0.02, batch_size=100, callbacks=[es_callback, cp_callback, csv_logger])

    # To be used after training
    def reload_existing_model(self):
        self.fetch_dataset()
        self.prepare_model()
        self.reload_if_possible()
        self.full_history = pd.read_csv(self.history_path, names = ["epoch", "loss", "val_loss"])
        return (self.model, self.full_history)