
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import pandas as pd

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

campaign = es.Campaign()

##########################
# Generate training data #
##########################

# number of inputs
D = 51

########################################
# choose the active subspace dimension #
########################################
d = 2

campaign = es.Campaign()
data = campaign.load_hdf5_data(file_path='covidsim_samples_only_cont.hdf5')
params = data['params']
samples = data['samples']
p_max = np.max(params, axis=0)
p_min = np.min(params, axis=0)
# normalize inputs to [-1, 1]
params = (params - 0.5 * (p_min + p_max)) / (0.5 * (p_max - p_min))
samples = samples[:, -1].reshape([-1, 1])

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
                        n_iter=10000, n_layers=4, n_neurons=n_neurons, test_frac = test_frac, 
                        batch_size = 64, lamb=0.0, standardize_X=False)
        
        #########################
        # Compute error metrics #
        #########################
        
        analysis = es.analysis.ANN_analysis(surrogate)
        rel_err_train, rel_err_test = analysis.get_errors(params, samples)
        err_ANN[r, n, 0] = rel_err_train
        err_ANN[r, n, 1] = rel_err_test
                
        #####################
        # train DAS network #
        #####################
        
        das_surrogate = es.methods.DAS_Surrogate()
        das_surrogate.train(params, samples, d, n_iter=10000, n_layers=4, n_neurons=n_neurons, 
                            test_frac = test_frac, batch_size = 64, standardize_X=False)
        
        #########################
        # Compute error metrics #
        #########################
        
        das_analysis = es.analysis.DAS_analysis(das_surrogate)
        rel_err_train, rel_err_test = das_analysis.get_errors(params, samples)
        err_DAS[r, n, 0] = rel_err_train
        err_DAS[r, n, 1] = rel_err_test

# store data
campaign = es.Campaign()
campaign.store_data_to_hdf5({'err_ANN' : err_ANN, 'err_DAS' : err_DAS}, file_path='errors.hdf5')
