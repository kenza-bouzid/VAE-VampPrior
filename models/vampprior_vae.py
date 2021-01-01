import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from utils.layers import GatedDense

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

class Encoder_VampPrior(tfkl.Layer):
    """Maps input X to latent vector Z, represents q(Z|X)

    Slightly differs from vanilla vae since it also has to be used for the vampprior

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
        original_dim=(28, 28, 1),
        latent_dim=40,
        intermediate_dim=300,
        activation=None,
        name="encoder",
        **kwargs
    ):
        super(Encoder_VampPrior, self).__init__(name=name, **kwargs)
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
        # No longer activitvity regularizer in the encoder
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
        original_dim=(28, 28, 1),
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
                original_dim,
                tfd.Bernoulli.logits,
                name="bernoulli_decoder",
            )
        else:
            self.pre_reconstruct_layer = tfkl.Dense(
                tfpl.IndependentNormal.params_size(original_dim),
                activation=None,
                name="pre_reconstruct_layer_continous"
            )
            self.reconstruct_layer = tfpl.IndependentNormal(
                original_dim,
                name="gaussian_decoder",
            )  

    def call(self, inputs):
        inputs = self.inputLayer(inputs)
        x = self.dense_first(inputs)
        x = self.dense_second(x)
        x = self.pre_reconstruct_layer(x)
        return self.reconstruct_layer(x)



class VAE_VampPrior(tfk.Model):
    """Combines the encoder and decoder into an end-to-end model for training.

    Uses the VampPrior idea (variational mixture of posteriors)

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
        n_pseudo_inputs=500,
        pseudo_input_type="generate",
        pseudo_inputs=None,
        pseudo_input_mean=None,
        pseudo_input_std=None,
        n_monte_carlo_samples=5,
        original_dim=(28, 28, 1),
        intermediate_dim=300,
        latent_dim=40,
        prior="standard_gaussian",
        input_type="binary",
        activation=None,
        name="autoencoder",
        **kwargs
    ):
        super(VAE_VampPrior, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.n_pseudo_inputs = n_pseudo_inputs
        self.pseudo_input_type = pseudo_input_type
        self.n_monte_carlo_samples = n_monte_carlo_samples

        if pseudo_input_type == "generate":
            if pseudo_input_mean is None:
                raise("If pseudo_input_type is 'generate' a mean has to be provided")
            if pseudo_input_std is None:
                raise("If pseudo_input_type is 'generate' a std has to be provided")

            self.pre_pseudo_input_generator = tfd.Independent(
                tfd.Normal(
                    loc=pseudo_input_mean,
                    scale=pseudo_input_std,
                )
            )

            self.pre_pseudo_inputs = self.pre_pseudo_input_generator.sample()

            self.pseudo_input_layer = tfkl.Dense(
                original_dim,
                activation=None,
            )
        elif pseudo_input_type == "data":
            if pseudo_inputs is None:
                raise("If pseudo_input_type is 'data' a data has to be provided")
            self.pseudo_inputs = pseudo_inputs
        else:
            raise("VampPrior requires either initialization with data or paramters to genrater pseudo inputs")


        self.encoder = Encoder_VampPrior(
            original_dim=original_dim,
            latent_dim=latent_dim,
            intermediate_dim=intermediate_dim
        )
        self.decoder = Decoder(original_dim=original_dim,
                               latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim, 
                               activation=activation,
                               input_type=input_type)

        # The employed prior is a mixture of posteriors, i.e. we use the encoder
        # n_pseudo_inputs times

    def call(self, inputs):
        z = self.encoder(inputs)

        # If the pre pseudo inputs are generated we have to create the new
        # pseudo inputs 
        if self.pseudo_input_type == "generate":
            pseudo_inputs = self.pseudo_input_layer(self.pre_pseudo_inputs)
        elif self.pseudo_input_type == "data":
            pseudo_inputs = self.pseudo_inputs

        # Create the Vamp_Prior based on the current encoder network
        batch_size = inputs.shape[0]

        # Create batch size copy of pseudo inputs
        pseudo_inputs_batched = tf.expand_dims(pseudo_inputs, axis=0)
        pseudo_inputs_batched = tf.repeat(pseudo_inputs_batched, batch_size, axis=0)

        print(inputs.shape)
        print(pseudo_inputs.shape)
        print(pseudo_inputs_batched.shape)
        pseudo_input_posteriors = self.encoder(pseudo_inputs)
        print(pseudo_input_posteriors)
        self.prior = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=1/self.n_pseudo_inputs * tf.ones(self.n_pseudo_inputs)
            ),
            components_distribution=self.encoder(pseudo_inputs),
            name="vamp_prior"
        )
        # batch_prior = tfd.Independent(
        #     batch_size*
        # )
        print("Hello")
        print(self.prior)
        print(z)


        # Calculate the KL loss using a monte_carlo sample
        z_sample = self.prior.sample(self.n_monte_carlo_samples)
        print(z_sample.shape)
        z_log_prob = tf.transpose(z.log_prob(tf.expand_dims(z_sample, axis=1)))
        print(z_log_prob.shape)
        prior_log_prob = self.prior.log_prob(z_sample)
        print(prior_log_prob.shape)
        # Mean over monte-carlo samples and batch size
        kl_loss_total = prior_log_prob - z_log_prob
        print(kl_loss_total.shape)
        kl_loss = tf.reduce_mean(kl_loss_total)##, axis=1)
        print(kl_loss.shape)
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
    