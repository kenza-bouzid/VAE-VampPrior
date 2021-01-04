import tensorflow as tf
from tensorflow.python.types.core import Value
import tensorflow_probability as tfp
import numpy as np
from enum import Enum
from utils.layers import GatedDense
from utils.pseudo_inputs import PseudoInputs
from models.vae import VAE
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


class Prior(Enum):
    STANDARD_GAUSSIAN = 0
    VAMPPRIOR = 1


class InputType(Enum):
    BINARY = 0


class Encoder(tfkl.Layer):
    """Maps input X to latent vector Z, represents q(Z|X)

    Takes an image X (of shape pixel_x, pixel_y, color_channel_depth) and maps it into
    a Normal distribution of the latent space Z (of shape latent_dim). It does so by
    calculating the mean and variance of the Gaussian (i.e. has a last deterministic layer
    of size 2*latent_dim).

    params
    ----

    prior: The tensorflow probability distribution representing
        prior the distribution on the latent space (i.e. p(Z))

    original_dim: Shape of the image given as (pixel_x, pixel_y, color_channel_depth)

    latent_dim: Shape of the latent space (in general rank-1 tensor, i.e. a vector),
        has to agree with the event shape of the prior distribution

    intermediate_dim: Number of hidden units in each hidden layer

    activation: The activation function to use in the gated dense layers
        (None by default because the Gated already used sigmoid)
    """

    def __init__(
        self,
        original_dim=(28, 28),
        latent_dim=40,
        intermediate_dim=300,
        activation=None,
        name="encoder",
        **kwargs
    ):
        super(Encoder, self).__init__(name=name, **kwargs)

        # Layers definitions
        self.input_layer = tfkl.InputLayer(
            input_shape=original_dim, name="input")
        self.flatten = tfkl.Flatten()
        self.dense_first = GatedDense(
            intermediate_dim, name="first_gated_encod", activation=activation)
        self.dense_second = GatedDense(
            intermediate_dim, name="second_gated_encod", activation=activation)

        # Need this layer to match the parameters of the normal distribution
        self.pre_latent_layer = tfkl.Dense(
            tfpl.IndependentNormal.params_size(latent_dim),
            activation=None,
            name="pre_latent_layer",
        )
        # Activity Regularizer adds the KL Divergence loss to the encoder
        self.latent_layer = tfpl.IndependentNormal(
            latent_dim,
            name="variational_encoder"
        )

    def call(self, inputs):
        inputs = self.input_layer(inputs)
        x = self.flatten(inputs)
        x = self.dense_first(x)
        x = self.dense_second(x)
        x = self.pre_latent_layer(x)
        return self.latent_layer(x)


class Decoder(tfkl.Layer):
    """Maps latent vector Z to the original shape X (e.g an image)

    Takes a sample from the latent space Z (of shape latent_dim) and maps it into
    a distribution over possible images X (of shape pixel_x, pixel_x, color_channel_depth)
    which correspond to the latent value. This output is either a Bernoulli for a binary
    color channel or a Normal for a continous color channel.

    If the output is Bernoulli then the last deterministic layer has to contain
    1*tf.prod(original_dim) parameters (since a Bernoulli only has one parameter
    per image pixel). If the output is normal then the last deterministic layer has
    to contain 2*tf.prod(original_dim) parameters (since a Normal has two parameters
    per pixel (mean and variance)).

    params
    ----

    original_dim: shape of the X space (given as the shape of a picture
        (pixel_x, pixel_y, color_channel_depth))

    latent_dim: Shape of the latent space (in general rank-1 tensor, i.e. a vector)

    intermediate_dim: Number of hidden units in each hidden layer

    activation: The activation function to use in the gated dense layers
        (None by default because the Gated already used sigmoid)

    input_type: In terms of image, whether the color channel is "binary" (i.e. black
        white) or "continous" (i.e. floating point value between 0.0 and 1.0)
    """

    def __init__(
        self,
        original_dim=(28, 28),
        latent_dim=40,
        intermediate_dim=300,
        activation=None,
        input_type=InputType.BINARY,
        name="decoder",
        **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.inputLayer = tfkl.InputLayer(
            input_shape=[latent_dim])
        self.dense_first = GatedDense(
            intermediate_dim, name="first_gated_decod", activation=activation)
        self.dense_second = GatedDense(
            intermediate_dim, name="first_gated_decod", activation=activation)
        if input_type == InputType.BINARY:
            self.pre_reconstruct_layer = tfkl.Dense(
                tfpl.IndependentBernoulli.params_size(original_dim),
                activation=None,
                name="pre_reconstruct_layer_binary"
            )
            self.reconstruct_layer = tfpl.IndependentBernoulli(
                original_dim, tfd.Bernoulli.logits)
        else:
            self.pre_reconstruct_layer = tfkl.Dense(
                tfpl.IndependentNormal.params_size(original_dim),
                activation=None,
                name="pre_reconstruct_layer_continous"
            )
            self.reconstruct_layer = tfpl.IndependentNormal(
                original_dim)  # do you know what we have to add here?

    def call(self, inputs):
        inputs = self.inputLayer(inputs)
        x = self.dense_first(inputs)
        x = self.dense_second(x)
        x = self.pre_reconstruct_layer(x)
        return self.reconstruct_layer(x)


class VanillaVAE(VAE):
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
        n_monte_carlo_samples=5,
        pseudo_inputs: PseudoInputs = None,
        kl_weight=3,
        activation=None,
        name="vanilla_vae",
        **kwargs
    ):
        super(VanillaVAE, self).__init__(
            original_dim,
            intermediate_dim,
            latent_dim, #? TODO : utile dans cette classe?
            prior_type,
            input_type,
            n_monte_carlo_samples,
            kl_weight,
            pseudo_inputs,
            activation,
            name)

        self.encoder = Encoder(original_dim=self.original_dim,
                               latent_dim=self.latent_dim,
                               intermediate_dim=self.intermediate_dim,
                               activation = self.activation)
        self.decoder = Decoder(original_dim=self.original_dim,
                               latent_dim=self.latent_dim,
                               intermediate_dim=self.intermediate_dim,
                               activation=self.activation,
                               input_type=self.input_type)

    def compute_kl_loss(self, z):
        # Calculate the KL loss using a monte_carlo sample
        z_sample = self.prior.sample(self.n_monte_carlo_samples)

        # Add additional dimension to enable broadcasting with the vamp prior,
        # then reverse because the batch_dim is required to be the first axis
        z_log_prob = tf.transpose(z.log_prob(tf.expand_dims(z_sample, axis=1)))
        # print(z_log_prob.shape)
        prior_log_prob = self.prior.log_prob(z_sample)
        # print(prior_log_prob.shape)
        # Mean over monte-carlo samples and batch size
        kl_loss_total = prior_log_prob - z_log_prob
        # print(kl_loss_total.shape)
        kl_loss = tf.reduce_mean(kl_loss_total)
        return self.kl_weight * kl_loss

    def call(self, inputs):
        z = self.encoder(inputs)

        if self.prior_type == Prior.VAMPPRIOR:
            self.recompute_prior()

        kl_loss = self.compute_kl_loss(z)
        self.add_loss(kl_loss)

        reconstructed = self.decoder(z)
        return reconstructed
