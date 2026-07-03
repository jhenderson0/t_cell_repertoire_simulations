import numpy as np
from functools import partial

import repertoire_simulator.lib as lib
import repertoire_simulator.models as models

#################################################################################
# Model setup
#################################################################################

# Simulation timing
t_start = 0
t_end = 10
dt = 0.01

# System dimensions
S = 1 # number of initial clones
R = S # number of initial antigens
N = 1 # number of patches

# initialisation 
c_initial = 0.0#np.ones((S, N)) *  np.random.zipf(2.2, size=S)[:, None] 
a_initial = 0.0

c_new = 1.0
a_new = 0.0
c_replace_cutoff = 0.1
c_local_cutoff = 0.1

# main parameter state dict
initial_param_state = {}

# clonal dynamic rates
alpha = 1.2
b = 1e7 #basal birth rate (years^-1) 
M = 1 #migration timescale (years^-1)

#all to all migration - will take to be homogenous for now
base_migration_matrix = np.full((N, N), M / N)
np.fill_diagonal(base_migration_matrix, 0.0)
migration_matrix = np.broadcast_to(base_migration_matrix[None, :, :], (1, N, N)).copy()

# clonal dynamics functions
initial_param_state['homeostatic_control'] = {'b': b, 'T0': 1}
homeostatic_control_func = models.local_homeostatic_competition_func

initial_param_state['antigen_response'] = {}
antigen_response_func = models.simple_antigen_response_func

initial_param_state['migration'] = {'migration_matrix': migration_matrix}
migration_func = models.simple_migration_func

# antigen dynamics 
lamb = 100 #antigen decay rate (years^-1)
D = 1 #antigenic environment strength (years^-1)
antigen_update_func = models.ou_antigen_update
initial_param_state['antigen'] = {'D': D, 'lamb': lamb}

demographic_stochasticity='yes'

# update method
continuum_update_method="euler" 

#################################################################################
# Run the simulation
#################################################################################
print("Starting repertoire simulation...")

ratios = np.logspace(3.5, 5.5, 10)
for ratio in ratios:
    print(f"Running sim for ratio = {ratio}")
    theta_c = D * ratio
    d = 0.2 * b / ratio
    
    initial_param_state['death'] = {'d': d}
    death_func = models.simple_death_func
    
    c, a, param_state, records = lib.simulate_repertoire(homeostatic_control_func=homeostatic_control_func,
                                                death_func=death_func,
                                                antigen_response_func=antigen_response_func,
                                                migration_func=migration_func,
                                                antigen_update_func=antigen_update_func,
                                                initial_param_state=initial_param_state,
                                                t_start=t_start, t_end=t_end, dt=dt,
                                                theta_c=theta_c, S=S, R=R, N=N,
                                                c_initial=c_initial, a_initial=a_initial, c_new=c_new, a_new=a_new,
                                                c_replace_cutoff=c_replace_cutoff, c_local_cutoff=c_local_cutoff,
                                                continuum_update_method=continuum_update_method,
                                                demographic_stochasticity=demographic_stochasticity, verbose=True, sample_dt=0.05)

    print("Simulation complete! Saving results...")
 
    c[c < 1] = 0.0

    np.savez_compressed(f"../../../../data/how_to_maintain_diversity/source_of_clones/ratio_{ratio}.npz", 
                        **{key: np.array(value, dtype=object) for key, value in records.items()})