import numpy as np
from functools import partial

import repertoire_simulator.lib as lib
import repertoire_simulator.models as models

#################################################################################
# Model setup
#################################################################################
#rng
seed = 100

# Simulation timing
t_start = 0
t_end = 1000
dt = 0.001

# System dimensions
S = 1 # number of initial clones
R = S # number of initial antigens
N = 1 # number of patches

# initialisation 
c_initial = 0.0 #np.ones((S, N)) *  np.random.zipf(2.2, size=S)[:, None] 
a_initial = 0.0

c_new = 1.0
a_new = 0.0
c_cutoff = 0.1

# main parameter state dict
initial_param_state = {}

# key parameters
D = 1 #antigenic environment strength (years^-1)
lamb = 1000 #antigen decay rate (years^-1)

alpha = 1.2 #power law exponent
gamma = alpha - 1 #ratio of recruitment to basal proliferation

M = 0.0 #migration timescale (years^-1) - not used

#all to all migration - will take to be homogenous for now
base_migration_matrix = np.full((N, N), M / N)
np.fill_diagonal(base_migration_matrix, 0.0)
migration_matrix = np.broadcast_to(base_migration_matrix[None, :, :], (1, N, N)).copy()

# clonal dynamics functions
d = D * (alpha + (alpha - 1) / gamma)
initial_param_state['death'] = {'d': d}
death_func = models.simple_death_func
    
initial_param_state['antigen_response'] = {}
antigen_response_func = models.simple_antigen_response_func

initial_param_state['migration'] = {'migration_matrix': migration_matrix}
migration_func = models.simple_migration_func

# antigen dynamics 
antigen_update_func = models.ou_antigen_update
initial_param_state['antigen'] = {'D': D, 'lamb': lamb}

demographic_stochasticity='yes'

# update method
continuum_update_method="euler" 

#################################################################################
# Run the simulation
#################################################################################
print("Starting repertoire simulation...")

ratios = [5e1] #np.logspace(1, 5.5, 10)
for i, ratio in enumerate(ratios):
    print(f"Running sim for ratio = {ratio}")
    
    theta_c = D * ratio
    b = theta_c / gamma
    
    initial_param_state['homeostatic_control'] = {'b': b, 'T0': 1}
    homeostatic_control_func = models.local_homeostatic_competition_func
    
    
    c, a, param_state, records = lib.simulate_repertoire(homeostatic_control_func=homeostatic_control_func,
                                                death_func=death_func,
                                                antigen_response_func=antigen_response_func,
                                                migration_func=migration_func,
                                                antigen_update_func=antigen_update_func,
                                                initial_param_state=initial_param_state,
                                                t_start=t_start, t_end=t_end, dt=dt,
                                                theta_c=theta_c, S=S, R=R, N=N,
                                                c_initial=c_initial, a_initial=a_initial, c_new=c_new, a_new=a_new,
                                                c_cutoff=c_cutoff,
                                                continuum_update_method=continuum_update_method,
                                                demographic_stochasticity=demographic_stochasticity, verbose=True, sample_dt=0.05, seed=seed+i)

    print("Simulation complete! Saving results...")
    
    np.savez_compressed(f"../../../../data/how_to_maintain_diversity/source_of_clones/long_time_sims/alpha_{alpha}_theta_{np.log10(ratio):.1f}.npz", **{key: np.array(value, dtype=object) for key, value in records.items()})