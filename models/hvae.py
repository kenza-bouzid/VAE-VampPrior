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

    def __init__(
        self,
        prior1,
        original_dim=(28, 28),
        intermediate_dim=300,
        latent_dim1=40,
        latent_dim2=40,
        activation=None,
        name="encoder",
        **kwargs
    ):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.prior1 = prior1
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
        self.latent_layer_z1 = tfpl.IndependentNormal(self.latent_dim1) #? , activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior1, use_exact_kl=True)

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

    def call(self, inputs):
        x = self.input_layer_x(inputs)
        x = self.flatten(x)
        z2 = self.forward_qz2_layers(x)
        z2 = self.input_layer_z2(z2)
        z1 = self.forward_qz1_layers(x, z2)
        return z1, z2


class Decoder(tfkl.Layer):

    def __init__(
        self,
        original_dim=(28, 28),
        latent_dim1=40,
        latent_dim2=40,
        intermediate_dim=300,
        input_type=InputType.BINARY,
        activation=None,
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

        if self.input_type == InputType.BINARY:
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
        z1 = self.forward_pz1(z2)
        generated = self.forward_px(z1, z2)
        return generated

    def forward_pz1(self, z2):
        z1_p = self.pz1_l1(z2)
        z1_p = self.pz1_l2(z1_p)
        z1_p = self.pre_latent_layer_pz1(z1_p)
        z1_p = self.p_z1(z1_p)
        return z1_p

    def forward_px(self, z1, z2):
        x1 = self.px_z1(z1)
        x2 = self.px_z2(z2)
        x1_x2 = tfkl.concatenate([x1, x2], axis=-1)
        x = self.pre_reconstruct_layer(x1_x2)
        return self.reconstruct_layer(x)

    def call(self, z1, z2):
        z1 = self.input_layer_z1(z1)
        z2 = self.input_layer_z2(z2)
        z1_p = self.forward_pz1(z2) #! TODO pb de comprehension
        x = self.forward_px(z1, z2)
        return x, z1_p


class HVAE(VAE):

    def __init__(
        self,
        latent_dim1=40,
        name="hvae",
        **kwargs
    ):
        super(HVAE, self).__init__(
            name=name, **kwargs)
        self.latent_dim1 = latent_dim1
        self.prior1 = tfd.Independent(tfd.Normal(
            loc=tf.zeros(self.latent_dim1),
            scale=1.0,
        ), reinterpreted_batch_ndims=1)

        self.encoder = Encoder(prior1=self.prior1,
                               original_dim=self.original_dim,
                               intermediate_dim=self.intermediate_dim,
                               latent_dim1=self.latent_dim1,
                               latent_dim2=self.latent_dim,
                               activation=self.activation)

        self.decoder = Decoder(original_dim=self.original_dim,
                               intermediate_dim=self.intermediate_dim,
                               latent_dim1=self.latent_dim1,
                               latent_dim2=self.latent_dim,
                               activation=self.activation,
                               input_type=self.input_type)

    def update_prior1(self, mean_z1, stddev_z1):
        # mean_z1 = tf.reduce_sum(mean_z1, axis = 0)
        # stddev_z1 = tf.reduce_sum(stddev_z1, axis = 0)
        self.prior1 = tfd.Independent(tfd.Normal(
            loc=mean_z1,
            scale=stddev_z1,
        ), reinterpreted_batch_ndims=1)

    def call(self, inputs):
        z1, z2 = self.encoder(inputs)

        if self.prior_type == Prior.VAMPPRIOR:
            self.recompute_prior()

        reconstructed, z1_p = self.decoder(z1, z2)
        self.prior1 = z1_p
        self.encoder.prior1 = self.prior1

        kl_loss = self.compute_kl_loss(z2)
        self.add_loss(kl_loss)
        kl_loss_prior1 = tfd.kl_divergence(
                z1,
                self.prior1,
            )
        self.add_loss(tf.reduce_mean(kl_loss_prior1))

        return reconstructed

    def get_priors(self):
        if self.prior_type == Prior.VAMPPRIOR:
            self.prior = self.recompute_prior()
        return self.prior1, self.prior

    def refresh_priors(self, x_test):
        if self.prior_type == Prior.VAMPPRIOR:
            self.prior = self.recompute_prior()
        z1, z2 = self.encoder(x_test)
        _, z1_p = self.decoder(z1, z2)
        self.prior1 = z1_p
        #self.update_prior1(mean_z1, stddev_z1)


    def marginal_log_likelihood_over_all_samples(self, x_test, n_samples=5000):
        # update priors to avoid tensorflow probability exceptions
        self.refresh_priors(x_test)
        ll = []
        for one_x in x_test:
            one_x = tf.expand_dims(one_x, axis=0)
            ll.append(self.marginal_log_likelihood_one_sample(
                one_x, n_samples, refresh_priors=False
            )
            )
        return ll

    def marginal_log_likelihood_one_sample(self, one_x, n_samples=5000, refresh_priors = True):
        if refresh_priors:
            self.refresh_priors(one_x)

        # For one sample the KL is identical
        z1, z2 = self.encoder(one_x)
        kl = self.compute_kl_loss(z2, n_samples = n_samples) + \
            tfd.kl_divergence(z1, self.prior1)

        # n_samples different reconstruction errors
        reconst_img_dist_n_samples_batched = self.decoder.generate_img(
            z2.sample(n_samples))
        reconst_error = reconst_img_dist_n_samples_batched.log_prob(one_x)

        full_error = reconst_error - kl
        return tf.reduce_logsumexp(full_error) - tf.math.log(float(n_samples))
