# %%
import runner as runner
runner_instance = runner.Runner(dataset_key=runner.Dataset.CALTECH, architecture=runner.Architecture.VANILLA, prior_configuration=runner.PriorConfiguration.SG, n_epochs=2,root="../exports")
runner_instance.run()
# %%
import runner as runner
runner_instance = runner.Runner(dataset_key=runner.Dataset.CALTECH, architecture=runner.Architecture.VANILLA, prior_configuration=runner.PriorConfiguration.SG, n_epochs=2,root="../exports")
runner_instance.reload_existing_model()

# %%
import matplotlib.pyplot as plt
model = runner_instance.model
prior = model.get_prior()
encoder = model.get_encoder()
decoder = model.get_decoder()
generated_img = decoder(prior.sample(1)).mean()

plt.imshow(generated_img[0])

# %%
import sys
sys.path.append('../')
import utils.datasets as data
calt = data.get_dataset(data.DatasetKey.CALTECH)
calt['X'].shape
np.reshape(calt['X'], (-1, 28, 28)).shape
# %%
import numpy as np
calt['X'].shape
plt.imshow(calt['X'][np.random.randint(5000)].reshape(28,28), cmap = "Greys")
# %%
np.random.randint(52000
)
# %%
plt.imshow(runner_instance.x_train[4])