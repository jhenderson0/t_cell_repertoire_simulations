import numpy as np
from functools import partial

import repertoire_simulator.lib as lib
import repertoire_simulator.models as models

#################################################################################
# Model setup
#################################################################################

# Simulation timing
t_start = 0
t_end = 5
dt = 0.01

# System dimensions
S = int(1e4) # number of initial clones
R = S # number of initial antigens
N = 1 # number of patches

# initialisation 
c_initial = np.random.zipf(2.2, (S, N)) #0.0
a_initial = 0.0

c_new = 1.0
a_new = 0.0
c_replace_cutoff = 0.1
c_local_cutoff = 1e-11

# clonal dynamic rates
theta_c = 1e4 #rate of new clones into the repertoire (years^-1)
b = 1e7 #basal birth rate (years^-1) 
d = 1 #basal death rate (years^-1)
M = 0.0 #migration timescale (years^-1)

# main parameter state dict
initial_param_state = {}

#all to all migration - will take to be homogenous for now
base_migration_matrix = np.full((N, N), M / N)
np.fill_diagonal(base_migration_matrix, 0.0)
migration_matrix = np.broadcast_to(base_migration_matrix[None, :, :], (1, N, N)).copy()

# clonal dynamics functions
initial_param_state['homeostatic_control'] = {'b': b, 'T0': 1}
homeostatic_control_func = models.local_homeostatic_competition_func

initial_param_state['death'] = {'d': d}
death_func = models.simple_death_func

initial_param_state['antigen_response'] = {'epsilon' : 1e-6}
antigen_response_func = models.self_inhibition_antigen_response_func #models.simple_antigen_response_func

initial_param_state['migration'] = {'migration_matrix': migration_matrix}
migration_func = models.simple_migration_func

demographic_stochasticity='no'

# antigen dynamics 
lamb = 1000 #antigen decay rate (years^-1)
D = 1 #antigen fluctuation timescale (years^-1)
initial_param_state['antigen'] = {'D': D, 'lamb': lamb}
antigen_update_func = models.ou_antigen_update

# encounter_rate = 0.1
# lamb = 30
# A = 50
# initial_param_state['antigen'] = {'A': A, 'lamb': lamb, 'encounter_rate': encounter_rate}
# antigen_update_func = models.shot_noise_antigen_update

# update method
continuum_update_method="euler" 

#################################################################################
# Run the simulation
#################################################################################
print("Starting repertoire simulation...")

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

np.savez(f"../../../data/sim_results/simple_sim.npz", clonal_abundances=c)
np.savez_compressed("../../../data/sim_results/simple_sim_record.npz", 
                    **{key: np.array(value, dtype=object) for key, value in records.items()})