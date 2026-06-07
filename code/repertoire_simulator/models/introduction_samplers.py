# To be implemented - need to store affinity matrix a sparse matrix for diverse repertoires 

import numpy as np

# ###################################################################################
# General function to update model parameters on the introduction of a new clone
# ####################################################################################
def update_param_state_on_introduction(param_state, S, N, R, clone_index, replace, param_update_funcs,
                                       fixed_antigen_pool=False, prng=np.random):

    for key, update_func in param_update_funcs.items():
        param_state[key] = update_func(param_state[key], clone_index=clone_index, replace=replace, N=N, prng=prng)

    return param_state

###################################################################################
#Antigen affinity updates 
####################################################################################

