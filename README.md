# VAE with a Variational Mixture of Posteriors "VampPrior"
This is an implementation of the following paper in Tensorflow 2.0:  
* Jakub M. Tomczak, Max Welling, VAE with a VampPrior, [arXiv preprint](https://arxiv.org/abs/1705.07120), 2017

We hereby compare the performance of a new prior ("Variational Mixture of Posteriors" prior, or VampPrior for short) for the variational auto-encoder framework with one layer and two layers of stochastic hidden units.

## Models
We provide a vanilla version of a VAE and a Hierarchical one with two level priors. Each architecture can be used with and without the VampPrior.  The VampPrior let you train the pseudo-inputs or randomly choose them from the data.
The models training can be monitored with both training and validation losses. The overall performance is quantified by the marginal test log likelihood.
You can run a vanilla VAE, a two-layered HVAE with the standard prior or the VampPrior by setting `architecture` argument to either: (i) `runner.Architecture.HVAE` or `architecture=runner.Architecture.VANILLA` for HVAE, (ii) and specifying `prior_configuration` argument to either `runner.PriorConfiguration.SG` or `runner.PriorConfiguration.VAMPGEN` or `PriorConfiguration.VAMPDATA`.


## Requirements
The code was implemeted with:
* `Tensorflow >= 2.0 `
* `Tensorflow Probability`

## Data
The experiments were conducted on the following datasets:
* static MNIST: links to the datasets can found at [link](https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST)
* OMNIGLOT: the dataset could be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat)
* Caltech 101 Silhouettes: the dataset could be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat)

You can specify the `dataset_key` argument to either `runner.Dataset.CALTECH` or `runner.Dataset.MNIST` or `runner.Dataset.OMNIGLOT` 

## Set an Experiment 

We provide a framework test for running the different experiments, refer to `experiments/test_runner.py` for a quick tutorial.

## References

```
@article{TW:2017,
  title={{VAE with a VampPrior}},
  author={Tomczak, Jakub M and Welling, Max},
  journal={arXiv},
  year={2017}
}
```