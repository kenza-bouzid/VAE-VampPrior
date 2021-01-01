import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from utils.layers import GatedDense

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
        prior1: tfp.distributions.Distribution,
        prior2: tfp.distributions.Distribution,
        original_dim=(28, 28),
        latent_dim1=40,
        latent_dim2=40,
        intermediate_dim=300,
        activation=None,
        name="encoder_qz2",
        **kwargs
    ):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.prior1 = prior1
        self.prior2 = prior2
        self.original_dim = original_dim
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.intermediate_dim = intermediate_dim
        self.activation = activation

        self.input_layer_x = tfkl.InputLayer(
            input_shape=self.original_dim, name="input_x")
        self.flatten = tfkl.Flatten()

        self.set_qz2_layers()

        self.input_layer_z2 = tfkl.InputLayer(
            input_shape=[self.latent_dim2], name="input_z2")

        self.set_qz1_layers()

    def set_qz2_layers(self):

        self.qz2_first = GatedDense(
            self.intermediate_dim, name="first_gated_encod_qz2", activation=self.activation)
        self.qz2_second = GatedDense(
            self.intermediate_dim, name="second_gated_encod_qz2", activation=self.activation)

        # Need this layer to match the parameters of the normal distribution
        self.pre_latent_layer_z2 = tfkl.Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim2),
            activation=None,
            name="pre_latent_layer_z2",
        )
        # Activity Regularizer adds the KL Divergence loss to the encoder
        self.latent_layer_z2 = tfpl.IndependentNormal(
            self.latent_dim2,
            activity_regularizer=tfpl.KLDivergenceRegularizer(
                self.prior2, use_exact_kl=True),
            name="latent_layer_z2")

    def set_qz1_layers(self):

        self.q_z1_x = GatedDense(
            self.intermediate_dim, name="first_gated_encod_qz1_x", activation=self.activation)

        self.q_z1_z2 = GatedDense(
            self.intermediate_dim, name="second_gated_encod_qz2_z1", activation=self.activation)

        self.q_z1_joint = GatedDense(
            2*self.intermediate_dim, name="second_gated_encod_qz2_joint", activation=self.activation)

        # Need this layer to match the parameters of the normal distribution
        self.pre_latent_layer_z1 = tfkl.Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim1),
            activation=None,
            name="pre_latent_layer_z1",
        )
        # Activity Regularizer adds the KL Divergence loss to the encoder
        self.latent_layer_z1 = tfpl.IndependentNormal(
            self.latent_dim1, activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior1, use_exact_kl=True))

    def forward_qz2_layers(self, x):
        z2 = self.qz2_first(x)
        z2 = self.qz2_second(z2)
        z2 = self.pre_latent_layer_z2(z2)
        return self.latent_layer_z2(z2)

    def forward_qz1_layers(self, x, z2):
        z1_x = self.q_z1_x(x)
        z1_z2 = self. q_z1_z2(z2)
        z1_x_z2 = tfkl.concatenate([z1_x, z1_z2], axis=1)
        z1_x_z2 = self.q_z1_joint(z1_x_z2)
        z1 = self.pre_latent_layer_z1(z1_x_z2)
        return self.latent_layer_z1(z1)

    def update_pz1(self, mean_z1, stddev_z1):
        self.prior1 = tfd.Independent(tfd.Normal(
            loc=mean_z1,
            scale=stddev_z1,
        ), reinterpreted_batch_ndims=1)

    def get_prior1(self):
        return self.prior1

    def call(self, inputs):
        x = self.input_layer_x(inputs)
        x = self.flatten(x)
        z2 = self.forward_qz2_layers(x)
        z2 = self.input_layer_z2(z2)
        z1 = self.forward_qz1_layers(x, z2)
        return z1, z2


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
        latent_dim1=40,
        latent_dim2=40,
        intermediate_dim=300,
        activation=None,
        input_type='binary',
        name="decoder",
        **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.input_type = input_type

        self.input_layer_z2 = tfkl.InputLayer(
            input_shape=[self.latent_dim2])

        self.set_pz1_layers()

        self.input_layer_z1 = tfkl.InputLayer(
            input_shape=[self.latent_dim1])

        self.set_px_layers()

    def set_pz1_layers(self):

        self.pz1_l1 = GatedDense(
            self.intermediate_dim, name="first_gated_decod_pz1", activation=self.activation)
        self.pz1_l2 = GatedDense(
            self.intermediate_dim, name="second_gated_decod_pz1", activation=self.activation)

        self.pre_latent_layer_pz1 = tfkl.Dense(
            tfpl.IndependentNormal.params_size(self.latent_dim1),
            activation=None,
            name="pre_latent_layer_pz1",
        )

        self.p_z1 = tfpl.IndependentNormal(
            self.latent_dim1)

    def set_px_layers(self):

        self.px_z1 = GatedDense(
            self.intermediate_dim, name="z1_gated_decod_px", activation=self.activation)
        self.px_z2 = GatedDense(
            self.intermediate_dim, name="z2_gated_decod_px", activation=self.activation)

        if self.input_type == 'binary':
            self.pre_reconstruct_layer = tfkl.Dense(
                tfpl.IndependentBernoulli.params_size(self.original_dim),
                activation=None,
                name="pre_reconstruct_layer_binary"
            )
            self.reconstruct_layer = tfpl.IndependentBernoulli(
                self.original_dim, tfd.Bernoulli.logits)
        else:
            self.pre_reconstruct_layer = tfkl.Dense(
                tfpl.IndependentNormal.params_size(self.original_dim),
                activation=None,
                name="pre_reconstruct_layer_continous"
            )
            self.reconstruct_layer = tfpl.IndependentNormal(
                self.original_dim)  # do you know what we have to add here?

    def get_pz1(self):
        return self.pz1

    def generate_img(self, z2):
        z1, _, _, = self.forward_pz1(z2)
        generated = self.forward_px(z1, z2)
        return generated

    def forward_pz1(self, z2):
        z1 = self.pz1_l1(z2)
        z1 = self.pz1_l2(z1)
        z1 = self.pre_latent_layer_pz1(z1)
        z1 = self.p_z1(z1)
        mean = z1.mean()
        stddev = z1.stddev()
        return z1, mean, stddev

    def forward_px(self, z1, z2):
        x1 = self.px_z1(z1)
        x2 = self.px_z2(z2)
        x1_x2 = tfkl.concatenate([x1, x2], axis=1)
        x = self.pre_reconstruct_layer(x1_x2)
        return self.reconstruct_layer(x)

    def call(self, z1, z2):
        z1 = self.input_layer_z1(z1)
        z2 = self.input_layer_z2(z2)
        _, mean, stddev = self.forward_pz1(z2)
        x = self.forward_px(z1, z2)
        return x, mean, stddev


class HVAE(tfk.Model):
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
        latent_dim1=40,
        latent_dim2=40,
        intermediate_dim=300,
        prior2_type="standard_gaussian",
        input_type="binary",
        activation=None,
        name="autoencoder",
        **kwargs
    ):
        super(HVAE, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.intermediate_dim = intermediate_dim
        self.input_type = input_type
        self.activation = activation

        if prior2_type == "standard_gaussian":
            self.prior2 = tfd.Independent(tfd.Normal(
                loc=tf.zeros(latent_dim2),
                scale=1.0,
            ), reinterpreted_batch_ndims=1)

        self.prior1 = tfd.Independent(tfd.Normal(
            loc=tf.zeros(latent_dim1),
            scale=1.0,
        ), reinterpreted_batch_ndims=1)

        self.encoder = Encoder(self.prior1,
                               self.prior2,
                               self.original_dim,
                               self.latent_dim1,
                               self.latent_dim2,
                               self.intermediate_dim)

        self.decoder = Decoder(self.original_dim,
                               self.latent_dim1,
                               self.latent_dim2,
                               self.intermediate_dim,
                               self.activation,
                               self.input_type)

    def call(self, inputs):
        z1, z2 = self.encoder(inputs)
        reconstructed, mean_z1, stddev_z1 = self.decoder(z1, z2)
        self.encoder.update_pz1(mean_z1, stddev_z1)
        self.prior1 = self.encoder.get_prior1()
        return reconstructed

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_prior(self):
        return self.prior1, self.prior2
