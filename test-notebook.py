# %% Hello World
import matplotlib.pyplot as plt
import importlib
import vanilla_vae as vae
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions

# %%
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
# %%
x_train.shape
# %%
importlib.reload(vae)
model = vae.VariationalAutoEncoder()


def neg_log_likelihood(x, rv_x):
    return -rv_x.log_prob(x)


model.compile(optimizer=tf.optimizers.Adam(1.0e-3), loss=neg_log_likelihood)

# %%
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
model.fit(x_train, x_train, epochs=10, batch_size=100, callbacks=[callback])
# %%
prior = model.get_prior()
encoder = model.get_encoder()
decoder = model.get_decoder()
# print(encoder.input)
# %%
generated_img = decoder(prior.sample(1)).sample()

plt.imshow(generated_img[0])
# %%
# Test Marginal Log-Likelihood (????)
reconst_test_dist = model(x_test)

ll = tf.reduce_mean(reconst_test_dist.log_prob(x_test))
ll
# %%
