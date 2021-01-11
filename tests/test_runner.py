# %%
import importlib
import runner as runner
runner_instance = runner.Runner(dataset_key=runner.Dataset.MNIST, architecture=runner.Architecture.HVAE, prior_configuration=runner.PriorConfiguration.SG, nb_epochs=1000)
runner_instance.run()
# %%
runner_instance.model.history.history