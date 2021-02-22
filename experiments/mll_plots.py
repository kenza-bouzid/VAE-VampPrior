# %%
import runner as runner
import seaborn as sns
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


def plot_mll(architecture):
    dict_clr = {0:'g', 1:'b', 2:'r'}
    for i, prior in enumerate(runner.prior_key_dict):
        file_name = f"MNIST_{architecture}_{runner.prior_key_dict[prior]}"
        mll = np.genfromtxt(f"../exports/mll/{file_name}.csv", delimiter=',')
        sns.kdeplot(mll, color=dict_clr[i], Label=f'MLL_{file_name}')
    plt.ylabel("marginal loglikelihood")
    plt.legend(loc='upper left')
    plt.savefig(f"../data/images/mll_MNIST_{architecture}.png")
    plt.show()
    

# %%
plot_mll("HVAE")
#%%
plot_mll("VANILLA")
