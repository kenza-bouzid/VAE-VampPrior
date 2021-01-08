import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from enum import Enum
from utils.layers import GatedDense
from utils.pseudo_inputs import PseudoInputs
from abc import ABC, abstractmethod

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


class Prior(Enum):
    STANDARD_GAUSSIAN = 0
    VAMPPRIOR = 1


class InputType(Enum):
    BINARY = 0


class VAE(tfk.Model, ABC):
    """Combines the encoder and decoder into an end-to-end model for training.

    params
    ----

    original_dim: shape of the X space (given as the shape of a picture
        (pixel_x, pixel_y, color_channel_depth))

    intermediate_dim: Number of hidden units in each hidden layer

    latent_dim: Shape of the latent space (in general rank-1 tensor, i.e. a vector)

    prior: 

    """

    def __init__(
        self,
        original_dim=(28, 28),
        intermediate_dim=300,
        latent_dim=40,
        prior_type=Prior.STANDARD_GAUSSIAN,
        input_type=InputType.BINARY,
        n_monte_carlo_samples=1,
        kl_weight=1.0,
        pseudo_inputs: PseudoInputs = None,
        activation=None,
        name="vae",
        **kwargs
    ):
        super(VAE, self).__init__(name=name, **kwargs)
        print(prior_type)
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.prior_type = prior_type
        self.input_type = input_type
        self.n_monte_carlo_samples = n_monte_carlo_samples
        self.kl_weight = kl_weight
        self.pseudo_inputs = pseudo_inputs
        self.activation = activation

        if prior_type == Prior.STANDARD_GAUSSIAN:
            self.prior = tfd.Independent(tfd.Normal(
                loc=tf.zeros(self.latent_dim),
                scale=1.0,
            ), reinterpreted_batch_ndims=1)

        self.encoder = None
        self.decoder = None

    def recompute_prior(self):

        # Uses the current version of the encoder (i.e. the current state of the
        # updated weights in the neural network) to find the encoded
        # representation of the pseudo inputs
        pseudo_input_encoded_posteriors = self.encoder(
            self.pseudo_inputs(None))
        # If in hvae then dicard z1
        if isinstance(pseudo_input_encoded_posteriors, tuple):
            pseudo_input_encoded_posteriors = pseudo_input_encoded_posteriors[1]

        self.prior = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=1.0/self.pseudo_inputs.get_n() * tf.ones(self.pseudo_inputs.get_n())
            ),
            components_distribution=pseudo_input_encoded_posteriors,
            name="vamp_prior"
        )
        return self.prior

    def compute_kl_loss(self, z, n_samples=None):
        if n_samples is None:
            n_samples = self.n_monte_carlo_samples

        if self.prior_type == Prior.VAMPPRIOR:
            self.recompute_prior()
            kl_loss = tfp.vi.monte_carlo_variational_loss(
                self.prior.log_prob,
                z,
                sample_size=n_samples,
            )
        else:
            kl_loss = tfd.kl_divergence(
                z,
                self.prior,
            )
        
        return tf.reduce_mean(self.kl_weight * kl_loss, axis=-1)

    @abstractmethod
    def call(self, inputs):
        pass

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_prior(self):
        if self.prior_type == Prior.VAMPPRIOR:
            self.recompute_prior()
        return self.prior

    def neg_log_likelihood(self, x, rv_x):
        return - rv_x.log_prob(x)

    def prepare(self):
        """Convenience function to compile the model
        """
        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=self.neg_log_likelihood)

    @abstractmethod
    def marginal_log_likelihood_one_sample(self, one_x, n_samples=5000, beta=1):
        pass

    def marginal_log_likelihood_over_all_samples(self, x_test, n_samples=5000, beta=1):
        ll = []
        for one_x in x_test:
            one_x = tf.expand_dims(one_x, axis=0)
            ll.append(self.marginal_log_likelihood_one_sample(
                one_x, n_samples, beta))
        return ll
