import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from enum import Enum
from utils.layers import GatedDense
from utils.pseudo_inputs import PseudoInputs
from models.vae import VAE, Prior, InputType
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers


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
        input_type=InputType.BINARY,
        activation=None,
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
        name="vanilla_vae",
        **kwargs
    ):
        super(VanillaVAE, self).__init__(
            name=name, **kwargs)

        self.encoder = Encoder(original_dim=self.original_dim,
                               latent_dim=self.latent_dim,
                               intermediate_dim=self.intermediate_dim,
                               activation=self.activation)
        self.decoder = Decoder(original_dim=self.original_dim,
                               latent_dim=self.latent_dim,
                               intermediate_dim=self.intermediate_dim,
                               activation=self.activation,
                               input_type=self.input_type)

    def call(self, inputs):
        z = self.encoder(inputs)

        kl_loss_weighted = self.compute_kl_loss(z)

        self.add_loss(kl_loss_weighted)

        reconstructed = self.decoder(z)
        return reconstructed
    
    def refresh_priors(self):
        if self.prior_type == Prior.VAMPPRIOR:
            self.recompute_prior()

    def marginal_log_likelihood_one_sample(self, one_x, n_samples=5000, refresh_prior=True):
        if refresh_prior:
            self.refresh_prior()

        enc_dist = self.encoder(one_x)

        kl_loss_weighted = self.compute_kl_loss(enc_dist, n_samples=n_samples)

        reconst_img_dist_n_samples_batched = self.decoder(
            enc_dist.sample(n_samples)
        )
        reconst_error = reconst_img_dist_n_samples_batched.log_prob(one_x)

        full_error = reconst_error + kl_loss_weighted
        return tf.reduce_logsumexp(full_error) - tf.math.log(float(n_samples))
    
    def marginal_log_likelihood_over_all_samples(self, x_test, n_samples=5000):
        ll = []
        self.refresh_priors()
        for one_x in x_test:
            one_x = tf.expand_dims(one_x, axis=0)
            ll.append(self.marginal_log_likelihood_one_sample(
                one_x, n_samples, refresh_prior=False
            )
            )
        return ll