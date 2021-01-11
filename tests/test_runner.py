# %%
import runner as runner
runner_instance = runner.Runner(dataset_key=runner.Dataset.MNIST, architecture=runner.Architecture.VANILLA, prior_configuration=runner.PriorConfiguration.SG, n_epochs=5)
runner_instance.run()
# %%
import runner as runner
runner_instance = runner.Runner(dataset_key=runner.Dataset.MNIST, architecture=runner.Architecture.VANILLA, prior_configuration=runner.PriorConfiguration.SG, n_epochs=5)
runner_instance.reload_existing_model()

# %%
import matplotlib.pyplot as plt
model = runner_instance.model
prior = model.get_prior()
encoder = model.get_encoder()
decoder = model.get_decoder()
generated_img = decoder(prior.sample(1)).mean()

plt.imshow(generated_img[0])