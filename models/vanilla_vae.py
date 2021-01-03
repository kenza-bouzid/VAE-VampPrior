import tensorflow as tf
from tensorflow.python.types.core import Value
import tensorflow_probability as tfp
import numpy as np

from utils.layers import GatedDense

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
        # TODO: For hierarchical VAE probably has to be done differently
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
        input_type='binary',
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
        if input_type == 'binary':
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
        original_dim=(28, 28),
        intermediate_dim=300,
        latent_dim=40,
        prior_type="standard_gaussian",
        input_type="binary",
        activation=None,
        name="autoencoder",
        # vamp prior args
        n_pseudo_inputs=500,
        pseudo_inputs_type="generate",
        pseudo_inputs=None,
        pseudo_inputs_mean=0.0,
        pseudo_inputs_std=0.01,
        n_monte_carlo_samples=5,
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.prior_type = prior_type
        self.original_dim = original_dim
        self.n_pseudo_inputs = n_pseudo_inputs
        self.pseudo_input_type = pseudo_inputs_type
        self.n_monte_carlo_samples = n_monte_carlo_samples


        if prior_type == "standard_gaussian":
            self.prior = tfd.Independent(tfd.Normal(
                loc=tf.zeros(latent_dim),
                scale=1.0,
            ), reinterpreted_batch_ndims=1)
        else:
            self.initialize_pseudo_inputs(n_pseudo_inputs, pseudo_inputs_type, pseudo_inputs, pseudo_inputs_mean, pseudo_inputs_std)
        self.encoder = Encoder(original_dim=original_dim,
                               latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                               latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim,
                               activation=activation,
                               input_type=input_type)

    def initialize_pseudo_inputs(self, n_pseudo_inputs,pseudo_input_type, pseudo_inputs,pseudo_inputs_mean, pseudo_inputs_std):
        if pseudo_input_type == "generate":
            self.pre_pseudo_inputs = tf.eye(n_pseudo_inputs)

            self.pseudo_input_layer = tfkl.Dense(
                np.prod(self.original_dim),
                kernel_initializer=tfk.initializers.RandomNormal(mean = pseudo_inputs_mean, stddev = pseudo_inputs_std),
                activation = "relu",
                name ="pseudo_input_layer"
            )
        elif pseudo_input_type == "data":
            if pseudo_inputs is None:
                raise ValueError("If pseudo_input_type is 'data' a data has to be provided")
            self.pseudo_inputs = pseudo_inputs
        else:
            raise ValueError("VampPrior requires either initialization with data or paramters to genrater pseudo inputs")
             
    def recompute_prior(self):
        # If the pre pseudo inputs are generated we have to create the new
        # pseudo inputs
        #
        # For the "data" vampprior the pseudo inputs are fixed over all epochs
        # and batches. In case of the "generate" vampprior an additional fully
        # connected layer maps a scalar [TODO: Really just scalar?] to an image
        # (e.g in case of mnist [] -> [28, 28, 1] )
        if self.pseudo_input_type == "generate":
            pseudo_inputs = self.pseudo_input_layer(self.pre_pseudo_inputs)
        elif self.pseudo_input_type == "data":
            pseudo_inputs = self.pseudo_inputs

        # print(inputs.shape)
        # print(pseudo_inputs.shape)
        # print(pseudo_inputs_batched.shape)

        # Uses the current version of the encoder (i.e. the current state of the
        # updated weights in the neural network) to find the encoded
        # representation of the pseudo inputs 
        pseudo_input_encoded_posteriors = self.encoder(pseudo_inputs)
        # print(pseudo_input_posteriors)
       
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=1.0/self.n_pseudo_inputs * tf.ones(self.n_pseudo_inputs)
            ),
            components_distribution=pseudo_input_encoded_posteriors,
            name="vamp_prior"
        )
    def call(self, inputs):
        z = self.encoder(inputs)

        # TODO: KL divergence loss has to be added here in case we are using hierarchical VAE
        # kl_loss = -0.5 * tf.reduce_mean(
        #     z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        # )
        # self.add_loss(kl_loss)
        # z_mean = z.mean()
        # z_log_var = np.log(z.variance())
        if self.prior_type != "standard_gaussian":
            prior = self.recompute_prior()
        else: 
            prior = self.prior
        # Calculate the KL loss using a monte_carlo sample
        z_sample = prior.sample(self.n_monte_carlo_samples)
        # print(z_sample.shape)

        # Add additional dimension to enable broadcasting with the vamp prior,
        # then reverse because the batch_dim is required to be the first axis
        z_log_prob = tf.transpose(z.log_prob(tf.expand_dims(z_sample, axis=1)))
        # print(z_log_prob.shape)
        prior_log_prob = prior.log_prob(z_sample)
        # print(prior_log_prob.shape)
        # Mean over monte-carlo samples and batch size
        kl_loss_total = prior_log_prob - z_log_prob
        # print(kl_loss_total.shape)
        kl_loss = tf.reduce_mean(kl_loss_total)##, axis=1)
        # print(kl_loss.shape)
        print(kl_loss)
        self.add_loss(kl_loss)

        reconstructed = self.decoder(z)
        return reconstructed

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_prior(self):
        return self.prior

    def prepare(self):
        """Convenience function to compile the model
        """
        def neg_log_likelihood(x, rv_x):
            return - rv_x.log_prob(x)

        self.compile(optimizer=tf.keras.optimizers.Adam(), loss=neg_log_likelihood)
    