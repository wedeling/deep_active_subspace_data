"""
Script that can be used to recompute the test / training errors. Will take several hours.
"""

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import pandas as pd

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

# number of neurons, replicas and training / test splits
n_neurons = 10
n_replicas = 100
n_test_fracs = 10

test_fracs = np.linspace(0.5, 0.1, n_test_fracs)
err_ANN = np.zeros([n_replicas, n_test_fracs, 2])
err_DAS = np.zeros([n_replicas, n_test_fracs, 2])

for r in range(n_replicas):

    for n, test_frac in enumerate(test_fracs):

        ##########################
        # Train an ANN surrogate #
        ##########################
    
        surrogate = es.methods.ANN_Surrogate()
        # train ANN. the input parameters are already scaled to [-1, 1], so no need to
        # standardize these
        surrogate.train(params, samples, 
                        n_iter=10000, n_layers=4, n_neurons=n_neurons, 
                        test_frac = test_frac, batch_size = 64, standardize_X=False,
                        standardize_y=True)

        #########################
        # Compute error metrics #
        #########################
        
        analysis = es.analysis.ANN_analysis(surrogate)
        rel_err_train, rel_err_test = analysis.get_errors(params, samples, relative=True)
        err_ANN[r, n, 0] = rel_err_train
        err_ANN[r, n, 1] = rel_err_test
        
        ########################################
        # choose the active subspace dimension #
        ########################################
        d = 1
        
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

# store errors
campaign = es.Campaign()
campaign.store_data_to_hdf5({'err_ANN' : err_ANN, 'err_DAS' : err_DAS}, file_path='errors.hdf5')
