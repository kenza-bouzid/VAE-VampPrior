# %%
import sys
sys.path.append('../')

import utils.evaluation as evaluation
import models.hvae as hvae
import datetime
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt
import models.vae as vae
import models.vanilla_vae as vanilla_vae
import models.hvae as hvae
import utils.datasets as datasets
import utils.pseudo_inputs as pseudo_inputs
from utils.datasets import DatasetKey, get_dataset
from utils.pseudo_inputs import PInputsData, PInputsGenerated, PseudoInputs
from dataclasses import dataclass

# %%
@dataclass
class ArgsRun():
    dataset_key : DatasetKey
    architecture : vae.VAE
    nb_epochs = 2000
    prior_type : vae.Prior
    pseudo_inputs_type : pseudo_inputs.PseudoInputs
    def fetch_dataset(self):
        (self.x_train, self.x_test) = datasets.get_dataset(DatasetKey.MNIST)

    def get_prepared_model(self):
        if self.prior_type == vae.Prior.VAMPPRIOR:

        self.model = self.architecture(original_dim = )

def run_and_save(args_model)
print(datasets.get_dataset(DatasetKey.MNIST))