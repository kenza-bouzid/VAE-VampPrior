import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
# class Sampling(layers.Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GatedDense(tfkl.Layer):
    """GatedDense layer subclass

    Use a combintation of two Dense layers with one activated and one not.
    Then component-wise multiply the results of both.
    """
    def __init__(self, size, name, activation=None):
        super(GatedDense, self).__init__(name=name, **kwargs)
        self.linear_1 = tfkl.Dense(size, kernel_initializer="he_normal")
        self.linear_2 = tfkl.Dense(size)
        self.activition = activation

    def call(self, inputs):

        x1 = self.linear_1(inputs)
        if self.activation:
            x1 = self.activation(x1)

        x2 = self.linear_2(inputs)
        x2 = tf.nn.sigmoid(x2)

        return tfkl.multiply([x1, x2])


class Encoder(layers.Layer):
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
        prior : tfp.distributions.Distribution,
        original_dim=(28, 28, 1),
        latent_dim=40,
        intermediate_dim=300,
        activation=None,
        name="encoder",
        **kwargs
        ):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.input = tfkl.InputLayer(shape = [28,28], name="input")
        self.flatten = tfkl.Flatten()
        self.dense_first = GatedDense(
            intermediate_dim, name="first_gated_encod", activation=activation)
        self.dense_second = GatedDense(
            intermediate_dim, name="second_gated_encod", activation=activation)
        # Need this layer to match the parameters of the normal distribution
        self.pre_latent_layer = tfkl.Dense(
            tfpl.IndependentNormal.param_size(latent_dim),
            activation=None,
            name="pre_latent_layer",
        )
        # Activity Regularizer adds the KL Divergence loss to the encoder
        # TODO: For hierarchical VAE probably has to be done differently
        self.latent_layer = tfpl.IndependantNormal(
            latent_dim, activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True))

    def call(self, inputs):
        inputs = self.input(inputs)
        x = self.flatten(inputs)
        x = self.dense_first(x)
        x = self.dense_second(x)
        x = self.pre_latent_layer(x)
        return self.latent_layer(x)


class Decoder(layers.Layer):
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
        original_dim=(28, 28, 1),
        latent_dim=40,
        intermediate_dim=300,
        activation=None,
        name="decoder",
        input_type='binary',
        **kwargs
        ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.inputLayer = tfkl.InputLayer(
            input_shape=[latent_dim])
        self.dense_first = GatedDense(
            intermediate_dim, name="first_gated_decod", activation=activation)
        self.dense_second = GatedDense(
            intermediate_dim, name="first_gated_decod", activation=activation)
        if input_type == 'binary':
            self.pre_reconstruct_layer = tfkl.Dense(
                tfpl.IndependentBernoulli.param_size(original_dim),
                activation=None,
                name="pre_reconstruct_layer_binary"
            )
            self.reconstruct_layer = tfpl.IndependentBernoulli(original_dim, tfd.Bernoulli.logits)
        else:
            self.pre_reconstruct_layer = tfkl.Dense(
                tfpl.IndependentNormal.param_size(original_dim),
                activation=None,
                name="pre_reconstruct_layer_continous"
            )
            self.reconstruct_layer = tfpl.IndependentNormal(original_dim) ##  do you know what we have to add here?

    def call(self, inputs):
        inputs = self.inputLayer(inputs)
        x = self.dense_first(inputs)
        x = self.dense_second(x)
        x = self.pre_reconstruct_layer(x)
        return self.reconstruct_layer(x)


class VariationalAutoEncoder(tfk.Model):
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
        original_dim=(28,28,1),
        intermediate_dim=300,
        latent_dim=40,
        prior="standard_gaussian",
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        if prior == "standard_gaussian":
            self.prior = tfd.Independent(tfd.Normal(
                loc=tf.zeros(latent_dim),
                scale=1.0,
            ))
        self.encoder = Encoder(self.prior,
                               original_dim=original_dim,
                               latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                               latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z = self.encoder(inputs)

        # TODO: KL divergence loss has to be added here in case we are using hierarchical VAE
        # kl_loss = -0.5 * tf.reduce_mean(
        #     z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        # )
        # self.add_loss(kl_loss)
        # z_mean = z.mean()
        # z_log_var = np.log(z.variance())

        reconstructed = self.decoder(z)
        return reconstructed
