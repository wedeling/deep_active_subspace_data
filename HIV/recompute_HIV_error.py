def draw():

    plt.clf()
    ax = fig.add_subplot(121)
    ax.set_ylim(bottom=0.001)
    ax.set_xlabel('training data size')
    ax.set_ylabel('relative error e [%]')
    ax.set_title('training error')
    sns.despine(top=True)
    offset=5
    
    # make the plot using all samples, not with confidence intervals
    
    ax.plot(data_size-offset, err_ANN_unconstrained[0:r+1,:,0].T * 100, '.', color='dodgerblue', label='unconstrained ANN')
    ax.plot(data_size, err_ANN[0:r+1,:,0].T * 100, '.', color='mediumaquamarine', label='constrained ANN')
    ax.plot(data_size+offset, err_DAS[0:r+1,:,0].T * 100, '.', color='salmon', label='DAS')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    ax2 = fig.add_subplot(122, sharey=ax)
    ax2.set_xlabel('training data size')
    ax2.set_title('test error')

    # make the plot using all samples, not with confidence intervals
    ax2.plot(data_size, err_ANN[0:r+1,:,1].T * 100, '.', color='mediumaquamarine')
    ax2.plot(data_size+offset, err_DAS[0:r+1,:,1].T * 100, '.', color='salmon')
    ax2.plot(data_size-offset, err_ANN_unconstrained[0:r+1,:,1].T * 100, '.', color='dodgerblue')

    sns.despine(left=True, ax=ax2)
    ax2.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.pause(0.1)

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import pandas as pd
import seaborn as sns
from collections import OrderedDict

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

fig = plt.figure(figsize=[8, 4])
offset=5

campaign = es.Campaign()

##########################
# Generate training data #
##########################

# number of inputs
D = 27
times = np.array([5, 15, 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 750,
                  1000, 1600, 1800, 2000, 2200, 2400, 2800, 3400])
T = times.size

# input - output data
data = campaign.load_hdf5_data(file_path='my_samples.hdf5')
params = data['inputs']
samples = data['outputs']

# time index
I = 5
samples = samples[:, I].reshape([-1, 1])

# number of neurons, replicas and tra
n_neurons = 100
n_replicas = 100
n_test_fracs = 10

err_ANN = np.zeros([n_replicas, n_test_fracs, 2])
err_ANN_unconstrained = np.zeros([n_replicas, n_test_fracs, 2])
err_DAS = np.zeros([n_replicas, n_test_fracs, 2])

# test fractions
test_fracs = np.linspace(0.9, 0.1, n_test_fracs)

# size of training data used
data_size = (1 - test_fracs) * samples.size

########################################
# choose the active subspace dimension #
########################################
d = 2

for r in range(n_replicas):

    for n, test_frac in enumerate(test_fracs):

        ########################################
        # Train an unconstrained ANN surrogate #
        ########################################
    
        surrogate_uc = es.methods.ANN_Surrogate()
        # train ANN. the input parameters are already scaled to [-1, 1], so no need to
        # standardize these
        surrogate_uc.train(params, samples, 
                        n_iter=10000, n_layers=4, n_neurons=n_neurons, 
                        test_frac = test_frac, batch_size = 64, 
                        standardize_X=False, standardize_y=True)

        #########################
        # Compute error metrics #
        #########################
        
        analysis = es.analysis.ANN_analysis(surrogate_uc)
        rel_err_train, rel_err_test = analysis.get_errors(params, samples, relative=True)
        err_ANN_unconstrained[r, n, 0] = rel_err_train
        err_ANN_unconstrained[r, n, 1] = rel_err_test

        ##########################
        # Train an ANN surrogate #
        ##########################
    
        surrogate = es.methods.ANN_Surrogate()
        # train ANN. the input parameters are already scaled to [-1, 1], so no need to
        # standardize these
        surrogate.train(params, samples, 
                        n_iter=10000, n_layers=4, n_neurons=[d, n_neurons, n_neurons], 
                        test_frac = test_frac, batch_size = 64, 
                        # activation = ['linear', 'tanh', 'tanh'],
                        bias = [False, True, True, True],
                        standardize_X=False, standardize_y=True)

        #########################
        # Compute error metrics #
        #########################
        
        analysis = es.analysis.ANN_analysis(surrogate)
        rel_err_train, rel_err_test = analysis.get_errors(params, samples, relative=True)
        err_ANN[r, n, 0] = rel_err_train
        err_ANN[r, n, 1] = rel_err_test

        
        #####################
        # train DAS network #
        #####################
        
        das_surrogate = es.methods.DAS_Surrogate()
        das_surrogate.train(params, samples, d, 
                            n_iter=10000, n_layers=4, n_neurons=n_neurons, 
                            test_frac = test_frac, batch_size = 64, standardize_X=False,
                            standardize_y=True)

        #########################
        # Compute error metrics #
        #########################
        
        das_analysis = es.analysis.DAS_analysis(das_surrogate)
        rel_err_train, rel_err_test = das_analysis.get_errors(params, samples, relative=True)
        err_DAS[r, n, 0] = rel_err_train
        err_DAS[r, n, 1] = rel_err_test
        
        draw()

# store data
campaign = es.Campaign()
campaign.store_data_to_hdf5({'err_ANN' : err_ANN, 'err_DAS' : err_DAS, 
                             'err_ANN_unconstrained' : err_ANN_unconstrained}, file_path='errors.hdf5')

