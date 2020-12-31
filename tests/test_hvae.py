# %% Hello World
"""
The following is for resolving paths for windows users 
(pretty sure linux handles it better)
Feel free to comment them out :)
"""
import sys
sys.path.append('../')

import models.hvae as hvae
import matplotlib.pyplot as plt
import importlib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions

# %% 
## Checking gpu setup
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# %%
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
# %%
x_train.shape
# %%
importlib.reload(hvae)
model = hvae.HVAE()

def neg_log_likelihood(x, rv_x):
    return -rv_x.log_prob(x)

model.compile(optimizer=tf.optimizers.Adam(1.0e-3), loss=neg_log_likelihood)

# %%
checkpoint_path = "../checkpoints/hvae/test.h5"
es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
with tf.device('/device:GPU:0'):
    model.fit(x_train, x_train, epochs=5,
              batch_size=100, callbacks=[es_callback, cp_callback])
# %%
model.load_weights(checkpoint_path)
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
n_samples = 20
prior_samples = prior.sample(n_samples)

posterior_dists_dependent = decoder(prior_samples)
posterior_dists = tfd.Independent(decoder(prior_samples), reinterpreted_batch_ndims=1)
print(posterior_dists_dependent)
print(posterior_dists)
ll = tf.reduce_mean(posterior_dists_dependent.log_prob(tf.reshape(x_test, (10000, 1, 28, 28))))
ll
# %%
